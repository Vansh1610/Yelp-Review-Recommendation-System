"""
Method Description:
For this competition, I used the model-based recommendation system method. 'More features lead to better model performance'; based on this line, I've built my logic to feature engineer new features to enhance model performance. From the user file, I've used the 'UserID' (for index), 'review count,' 'helpful,' 'fans,' 'friends count,' and total compliment score ( sum of all compliment columns for a row) to show the reliability of the user, and 'yelping since' is a feature that shows for long they have been rating business which offers their experience in the field. I've used the 'business' (for index) and 'is_open' from the business file to know if it's still in business and the total review count. To provide more features to the model, I've used the attributes columns in the business file as this provides more information about the business, like food quality, ambiance, etc., and these attributes could influence a user's rating. Each feature was preprocessed; if it was a boolean feature, it was assigned 1 or 0; label encoding if it was a categorical feature; and for numerical, it was converted to int or float. I used mean imputation to fill in the missing or None values, especially in numerical, to avoid errors. Once all the features were extracted, I combined all of the features using the index with train data, standardized some of the features, and trained the XgBoost model on the data. To improve accuracy, I increased the estimators and balanced them with a higher learning rate for faster convergence. I tried combining an item-based recommendation system with a model-based one to form a hybrid approach, but the item-based one had an RMSE greater than one, reducing the hybrid system's performance. I even tried increasing the feature count, but this added more complexity with no improvement.

Error Distribution:
>=0 and <1: 102236
>=1 and <2: 32869
>=2 and <3: 6140
>=3 and <4: 796
>=4: 3

RMSE:
0.97927903460538


Execution time:
917.98s
"""


from pyspark import SparkContext
import json
import sys
import time
import pandas as pd
import xgboost as xg
from sklearn.preprocessing import StandardScaler
import datetime
import pickle

start = time.time()


def error_dist(pred, gt):
    ans = {'>=0 and <1': 0, '>=1 and <2': 0,
           '>=2 and <3': 0, '>=3 and <4': 0, '>=4': 0}
    for k, v in pred.items():
        diff = abs(v-gt[k])
        if diff >= 0 and diff < 1:
            ans['>=0 and <1'] += 1
        elif diff >= 1 and diff < 2:
            ans['>=1 and <2'] += 1
        elif diff >= 2 and diff < 3:
            ans['>=2 and <3'] += 1
        elif diff >= 3 and diff < 4:
            ans['>=3 and <4'] += 1
        else:
            ans['>=4'] += 1

    return ans


def noise(x):
    s = {'average': 2, 'loud': 1, 'quiet': 3, 'very_loud': 0}

    if x is not None:
        if 'NoiseLevel' in x.keys():
            return s[x['NoiseLevel']]
        else:
            return None


def groups(x):
    if x is None:
        return 0
    else:
        if 'RestaurantsGoodForGroups' in x.keys():
            if x['RestaurantsGoodForGroups'] == 'True':

                return 1
            else:
                return 0
        else:
            return 0


def outdoor(x):

    if x is None:
        return 0
    else:
        if 'OutdoorSeating' in x.keys():
            if x['OutdoorSeating'] == 'True':

                return 1
            else:
                return 0
        else:

            return 0


def cards(x):

    if x is None:
        return 0
    else:
        if 'BusinessAcceptsCreditCards' in x.keys():
            if x['BusinessAcceptsCreditCards'] == 'True':

                return 1
            else:
                return 0
        else:

            return 0


def tv(x):

    if x is None:
        return 0
    else:
        if 'HasTV' in x.keys():
            if x['HasTV'] == 'True':

                return 1
            else:
                return 0
        else:

            return 0


def take(x):
    if x is None:
        return 0
    else:
        if 'RestaurantsTakeOut' in x.keys():
            if x['RestaurantsTakeOut'] == 'True':
                return 1

        elif 'RestaurantsDelivery' in x.keys():
            if x['RestaurantsDelivery'] == 'True':
                return 1

        return 0


def total_attributes(x):
    if x is not None:

        c = len(x.keys())
        for k, v in x.items():
            if v == 'False' or v == False or v == 'none':
                c -= 1
        return c
    else:
        return None


def good_kids(x):
    if x is None:
        return 0
    else:
        if 'GoodForKids' in x.keys():
            if x['GoodForKids'] == 'True':
                return 1
            else:
                return 0
        else:
            return 0


def price_range(x):
    if x is None:
        return 0
    else:
        if 'RestaurantsPriceRange2' in x.keys():
            return int(x['RestaurantsPriceRange2'])
        else:
            return None


def count_days(x):
    if x is None:
        return None
    return len(x.keys())


def bike_park(x):
    if x is None:
        return 0
    else:
        if 'BikeParking' in x.keys():
            if x['BikeParking'] == 'True':
                return 1
            else:
                return 0
        else:
            return 0


def wifi(x):
    if x is None:
        return 0
    else:
        if 'WiFi' in x.keys():
            if x['WiFi'] == 'True':
                return 1
            else:
                return 0
        else:
            return 0


def avg_hrs(x):
    if x is None:
        return None
    t = 0
    for i in x.keys():
        v = x[i]
        start = v.split('-')[0]
        end = v.split('-')[1]
        start = datetime.datetime.strptime(start, "%H:%M")
        end = datetime.datetime.strptime(end, "%H:%M")
        if end < start:

            start = datetime.datetime(2023, 1, 1, start.hour, start.minute)
            end = datetime.datetime(2023, 1, 2, end.hour, end.minute)
        t += (end-start).total_seconds()/3600
    return t/len(x.keys())


def is_weekend(x):
    if x is None:
        return 0
    else:
        if 'Saturday' or 'Sunday' in x.keys():
            return 1
        else:
            return 0


sc = SparkContext.getOrCreate()

sc.setLogLevel("WARN")

sc.setLogLevel("WARN")


folder_path = sys.argv[1]
val_file = sys.argv[2]
op_file = sys.argv[3]


rdd = sc.textFile(folder_path+'/yelp_train.csv').map(lambda x: x.split(','))

first_row = rdd.first()
rdd = rdd.filter(lambda x: x != first_row)
rdd = rdd.map(lambda x: (x[0], (x[1], float(x[2]))))


def convert_to_years(x):
    x = x.split('-')[0]
    return 2023-int(x)


def convert_categories(x):
    if x is None:
        return 0
    return len(x.split(','))


business = sc.textFile(folder_path+'/business.json').map(json.loads)
business = business.map(lambda x: (x['business_id'], (x['stars'], x['review_count'], x['is_open'], convert_categories(x['categories']), count_days(x['hours']), avg_hrs(x['hours']), is_weekend(x['hours']), bike_park(x['attributes']), wifi(
    x['attributes']), total_attributes(x['attributes']), price_range(x['attributes']), good_kids(x['attributes']), take(x['attributes']), noise(x['attributes']), outdoor(x['attributes']), groups(x['attributes']), cards(x['attributes']), tv(x['attributes']))))


def friends(x):
    if x is None:
        return 0
    else:
        return len(x.split(','))


def compliment(x):
    return x["compliment_hot"]+x["compliment_more"]+x["compliment_profile"]+x["compliment_cute"]+x["compliment_list"]+x["compliment_note"]+x["compliment_plain"]+x["compliment_cool"]+x["compliment_funny"]+x["compliment_writer"]+x["compliment_photos"]


user = sc.textFile(folder_path+'/user.json').map(json.loads)
user = user.map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'], convert_to_years(
    x['yelping_since']), x['useful'], x['fans'], friends(x['friends']), compliment(x))))
rdd_user = rdd.leftOuterJoin(user)

rdd_user = rdd_user.map(lambda x: (x[1][0][0], (x[0], x[1][1][0], x[1][1][1],
                        x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][0][1])))
business_rdd_user = rdd_user.leftOuterJoin(business)
business_rdd_user = business_rdd_user.map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7], x[1][0][8], x[1][1][0], x[1][1][1], x[1]
                                          [1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10], x[1][1][11], x[1][1][12], x[1][1][13], x[1][1][14], x[1][1][15], x[1][1][16], x[1][1][17]))

col = ['business_id', 'user_id', 'review_count_u', 'average_stars_users', 'yelping_since', 'useful', 'fans', 'friends', 'compliments', 'ratings', 'business_stars', 'review_count_b',
       'is_open', 'categories', 'days', 'avg_hrs', 'weekend', 'bike_park', 'wifi', 'tot_attr', 'price_range', 'good_kids', 'take_out', 'noise', 'outdoor', 'groups', 'cards', 'tv']


df = pd.DataFrame(business_rdd_user.collect(), columns=col)
mean_day = df['days'].mean()
mean_avg_hr = df['avg_hrs'].mean()
mean_attr = df['tot_attr'].mean()
mean_price = df['price_range'].mean()
df['days'] = df['days'].fillna(df['days'].mean())
df['avg_hrs'] = df['avg_hrs'].fillna(df['avg_hrs'].mean())
df['tot_attr'] = df['tot_attr'].fillna(df['tot_attr'].mean())
df['price_range'] = df['price_range'].fillna(int(df['price_range'].mean()))


mean_noise = int(df['noise'].mean())

df['noise'] = df['noise'].fillna(mean_noise)

x_train = df.drop(['business_id', 'user_id', 'ratings'], axis=1)
y_train = df['ratings']


mean_dict = {

    'days': mean_day, 'avg_hrs': mean_avg_hr, 'tot_attr': mean_attr, 'price_range': mean_price, 'noise': mean_noise

}

with open('mean.json', 'w') as f:
    json.dump(mean_dict, f)


ss = StandardScaler()
X_scaled = ss.fit_transform(
    x_train[['useful', 'fans', 'review_count_u', 'review_count_b', 'friends', 'compliments']])
X_scaled = pd.DataFrame(X_scaled, columns=[
                        'useful', 'fans', 'review_count_u', 'review_count_b', 'friends', 'compliments'])
X = x_train.drop(['useful', 'fans', 'review_count_u',
                 'review_count_b', 'friends', 'compliments'], axis=1)
X_scaled_train = pd.concat([X, X_scaled], axis=1)

model = xg.XGBRegressor(
    eval_metric="rmse", n_estimators=2000, learning_rate=0.1)
model.fit(X_scaled_train, y_train)
y_pred_train = model.predict(X_scaled_train)


with open('model_based.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(ss, f)


rdd_val = sc.textFile(val_file)
rdd_val = rdd_val.map(lambda x: x.split(','))
first_row = rdd_val.first()
rdd_val = rdd_val.filter(lambda x: x != first_row)


rdd_val = rdd_val.map(lambda x: (x[0], (x[1],)))


rdd_val_user = rdd_val.leftOuterJoin(user)


rdd_val_user = rdd_val_user.map(lambda x: (
    x[1][0][0], (x[0], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6])))

business_rdd_val_user = rdd_val_user.leftOuterJoin(business)


val_cols = ['business_id', 'user_id', 'review_count_u', 'average_stars_users', 'yelping_since', 'useful', 'fans', 'friends', 'compliments', 'business_stars', 'review_count_b',
            'is_open', 'categories', 'days', 'avg_hrs', 'weekend', 'bike_park', 'wifi', 'tot_attr', 'price_range', 'good_kids', 'take_out', 'noise', 'outdoor', 'groups', 'cards', 'tv']

business_rdd_val_user = business_rdd_val_user.map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7], x[1][1][0], x[1][1][1], x[1][1]
                                                  [2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10], x[1][1][11], x[1][1][12], x[1][1][13], x[1][1][14], x[1][1][15], x[1][1][16], x[1][1][17]))

df_val = pd.DataFrame(business_rdd_val_user.collect(), columns=val_cols)
df_val['days'] = df_val['days'].fillna(mean_day)
df_val['avg_hrs'] = df_val['avg_hrs'].fillna(mean_avg_hr)
df_val['tot_attr'] = df_val['tot_attr'].fillna(mean_attr)
df_val['price_range'] = df_val['price_range'].fillna(mean_price)


df_val['noise'] = df_val['noise'].fillna(mean_noise)

x_val = df_val.drop(['business_id', 'user_id'], axis=1)

X_scaled_val = ss.transform(
    x_val[['useful', 'fans', 'review_count_u', 'review_count_b', 'friends', 'compliments']])
X_scaled_val = pd.DataFrame(X_scaled_val, columns=[
                            'useful', 'fans', 'review_count_u', 'review_count_b', 'friends', 'compliments'])
X_val = x_val.drop(['useful', 'fans', 'review_count_u',
                   'review_count_b', 'friends', 'compliments'], axis=1)
X_scaled_val = pd.concat([X_val, X_scaled_val], axis=1)

y_pred_val = model.predict(X_scaled_val)


df_val = df_val.reset_index().drop('index', axis=1)
model_based_dict = {}
for i in range(len(y_pred_val)):
    model_based_dict[(df_val.iloc[i]['user_id'], df_val.iloc[i]
                      ['business_id'])] = y_pred_val[i]


with open(op_file, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in model_based_dict.keys():
        f.write(i[0]+','+i[1]+','+str(model_based_dict[i]))
        f.write('\n')
f.close()
