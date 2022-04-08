import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
###############################################################
#GOREV 1
###############################################################
# Adım 1
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_ = pd.read_csv("FLO/flo_data_20K.csv")

# Adım 2

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

# Adım 3

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# Adım 4

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Adım 5

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df.dtypes

###############################################################
#GOREV 2
###############################################################
#Adım 1
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
# Adım 2
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df['recency_cltv_weekly'] = ((df['last_order_date'] - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['T_weekly'] = ((today_date - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['frequency'] = df['order_num_total_ever']
cltv_df['monetary_cltv_avg'] = df['customer_value_total_ever'] / df['order_num_total_ever']
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

###############################################################
#GOREV 3
###############################################################
#Adım 1

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])


bgf.predict(4 * 3,
                cltv_df['frequency'],
                cltv_df['recency_cltv_weekly'],
                cltv_df['T_weekly']).sort_values(ascending=False).head(10)
bgf.predict(4 * 6,
                cltv_df['frequency'],
                cltv_df['recency_cltv_weekly'],
                cltv_df['T_weekly']).sort_values(ascending=False).head(10)

#3 aylık ve 6 aylık süreçteki satın alım tahminlere bakıldığında,6 aylık süreçte yalnızca
#miktarların değiştiği,kullanıcıların aynı kaldığı gözlenmiştir.
#Süreyle orantılı olarak miktarlar da iki katına çıkmıştır.

#Adım 2

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg'])

# Adım 3

cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
scaler = MinMaxScaler(feature_range=(0,1))
cltv_df["scaled_cltv"] = scaler.fit_transform(cltv_df[["cltv"]])

cltv_df.nlargest(n=20, columns=["cltv"])

###############################################################
#GOREV 4
###############################################################
# Adım 1
cltv_df["segment"] = pd.qcut(cltv_df["scaled_cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})

# Adım 2

#cltv verileri incelendiğinde  A grubunun diğer gruplara oranla ezici farkla üstün olduğu açıkça görünmektedir.
#Az sayıda segmente ayrılan data'larda segmentler arasındaki farklar istenilen düzeyde fark edilmemektedir.
#Bu sebepten fazla sayıda segmentin olması ile daha doğru gruplar elde edilebilir.
#Bu mantıkla gidildiğinde daha spesifik segmentler elde edilmiş olur ve bu gruplara göre aksiyon alınır.

# Adım 3

# Segment A
# Bu segmentteki müşteriler firmaya en çok getiri sağlama potansiyelindeki müşterilerdir. Bu müşteriler için belirli
# alışveriş tutarı geçildiğinde indirim verilebilir. Online alışverişte aylık olarak belirli miktarda alışveriş yapılması durumunda kargo
# ücreti alınmaması ve müşterilerin psikolojik olarak kendilerini üstün görmesi için elmas üye vs. isimlendirmeler verilmesi
# mantıklı olabilir. Ayrıca kendilerini referans alarak kattıkları üyelerin segmentlerine bakılıp, yeni kullanıcı hangi
# segmentteyse ona göre kuponlar tanımlanması yapılabilir. Böylece her müşteri bu segmentte kalmayı isteyecek ve alışveriş
# alışkanlıklarını devam ettirecektir.

# Segment D
# Bu segmentteki müşteriler müşteri yaşına bakıldığında en büyük değere sahiptir. Bu veri göz önüne alındığında
# görülmektedir ki bu müşteriler bu firmayı çok uzun zaman önce tercih etmişler fakat artık eskisi kadar tercih
# etmemektedir. Bu müşterilerin yeniden kazanılması için belirli bir yıl belirlenip, bu yıl kadar önce firmadan alışveriş
# yapan kişiler için kampanyalar düzenlenebilir. Ek olarak bu müşteriler belirli bir miktar üzerinde harcama yaparsa
# bir çekilişe katılabilir. Böylece müşteri yaşı uzun kişiler bu fırsatları görüp firmayı yeniden tercih edebilecektir.
