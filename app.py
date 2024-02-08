from flask import Flask, render_template, jsonify
import json
from pyspark import SparkContext
import sys
import time
import pandas as pd
import xgboost as xg
from sklearn.preprocessing import StandardScaler
import datetime
import pickle
import random

app = Flask(__name__)

with open('standard_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

with open('model_based.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


with open('mean.json', 'r') as f:
    means = json.load(f)

with open('reduced_data/business.json', 'r') as f:
    businesses = json.load(f)
    businesses = [{'id': i+1, 'business': businesses[i]}
                  for i in range(0, len(businesses))]

with open('reduced_data/user.json', 'r') as f:
    users_list = json.load(f)
    users = [{'id': i+1, 'user': users_list[i]}
             for i in range(0, len(users_list))]


with open('reduced_data/user_reviews.json', 'r') as f:
    user_reviews = json.load(f)

with open('reduced_data/business_reviews.json', 'r') as f:
    business_reviews = json.load(f)


def convert_to_years(x):
    x = x.split('-')[0]
    return 2023-int(x)


def convert_categories(x):
    if x is None:
        return 0
    return len(x.split(','))


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


def friends(x):
    if x is None:
        return 0
    else:
        return len(x.split(','))


def compliment(x):
    return x["compliment_hot"]+x["compliment_more"]+x["compliment_profile"]+x["compliment_cute"]+x["compliment_list"]+x["compliment_note"]+x["compliment_plain"]+x["compliment_cool"]+x["compliment_funny"]+x["compliment_writer"]+x["compliment_photos"]


@app.route('/')
def index():
    return render_template('home.html', users=users)


@app.route('/api/get_businesses/')
def get_businesses():
    selected_businesses = [business for business in businesses]
    return jsonify(selected_businesses)


@app.route('/predict/<user_id>/<business_id>')
def predict(user_id, business_id):
    user_data = users[int(user_id)-1]
    business_data = businesses[int(business_id)-1]

    user_data = user_data['user']

    business_data = business_data['business']
    user_sel = users[int(user_id)-1]
    business_sel = businesses[int(business_id)-1]

    business_data = (business_data['business_id'], business_data['stars'], business_data['review_count'], business_data['is_open'], convert_categories(business_data['categories']), count_days(business_data['hours']), avg_hrs(business_data['hours']), is_weekend(business_data['hours']), bike_park(business_data['attributes']), wifi(business_data['attributes']), total_attributes(
        business_data['attributes']), price_range(business_data['attributes']), good_kids(business_data['attributes']), take(business_data['attributes']), noise(business_data['attributes']), outdoor(business_data['attributes']), groups(business_data['attributes']), cards(business_data['attributes']), tv(business_data['attributes']))
    user_data = (user_data['user_id'], user_data['review_count'], user_data['average_stars'], convert_to_years(
        user_data['yelping_since']), user_data['useful'], user_data['fans'], friends(user_data['friends']), compliment(user_data))

    cols = ['business_id', 'user_id', 'review_count_u', 'average_stars_users', 'yelping_since', 'useful', 'fans', 'friends', 'compliments', 'business_stars', 'review_count_b',
            'is_open', 'categories', 'days', 'avg_hrs', 'weekend', 'bike_park', 'wifi', 'tot_attr', 'price_range', 'good_kids', 'take_out', 'noise', 'outdoor', 'groups', 'cards', 'tv']
    data = [business_data[0], user_data[0], user_data[1], user_data[2], user_data[3], user_data[4], user_data[5], user_data[6], user_data[7], business_data[1], business_data[2], business_data[3], business_data[4], business_data[5],
            business_data[6], business_data[7], business_data[8], business_data[9], business_data[10], business_data[11], business_data[12], business_data[13], business_data[14], business_data[15], business_data[16], business_data[17], business_data[18]]
    df = pd.DataFrame(data=[data], columns=cols)

    df['days'] = df['days'].fillna(means['days'])
    df['avg_hrs'] = df['avg_hrs'].fillna(means['avg_hrs'])
    df['tot_attr'] = df['tot_attr'].fillna(means['tot_attr'])
    df['price_range'] = df['price_range'].fillna(int(means['price_range']))
    df['noise'] = df['noise'].fillna(means['noise'])

    df_scaled = loaded_scaler.transform(
        df[['useful', 'fans', 'review_count_u', 'review_count_b', 'friends', 'compliments']])
    df_scaled = pd.DataFrame(df_scaled, columns=[
                             'useful', 'fans', 'review_count_u', 'review_count_b', 'friends', 'compliments'])
    df = df.drop(['useful', 'fans', 'review_count_u',
                 'review_count_b', 'friends', 'compliments'], axis=1)
    df_scaled = pd.concat([df, df_scaled], axis=1)
    pred = model.predict(df_scaled.iloc[:, 2:])[0]
    pred = min(pred, 5)

    pred = json.dumps({"result": round(float(pred), 1)})
    user_review_final = []
    for i in user_reviews:
        if i['user_id'] == user_sel['user']['user_id']:
            user_review_final.append(i['text'])

    business_review_final = []
    for i in business_reviews:
        if i['business_id'] == business_sel['business']['business_id']:
            business_review_final.append(i['text'])

    print(business_sel)
    print(business_review_final)

    if len(user_review_final) < 5:
        user_review_final = user_review_final
    else:
        user_review_final = random.sample(user_review_final, 5)

    if len(business_review_final) < 5:
        business_review_final = business_review_final
    else:
        business_review_final = random.sample(business_review_final, 5)

    return render_template('predict.html', result=pred, user=user_sel, business=business_sel, user_review_final=user_review_final, business_review_final=business_review_final)


if __name__ == '__main__':
    app.run(debug=True)
