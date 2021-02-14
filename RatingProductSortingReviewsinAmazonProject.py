######################################################################################
#             Rating Product & Sorting Reviews in Amazon Project                     #
######################################################################################

#<<<Şule AKÇAY>>>


#kütüphaneleri import ettim

import pandas as pd
import pymysql
import ast
import math
import scipy.stats as st


from sqlalchemy import create_engine
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

##############Features#################
# - Kullanıcı ID’si. Örn: A2SUAM1J3GNN3B
#asin – Ürün ID’si. Örn: 0000013714
#reviewerName – Kullanıcı Adı
#helpful – Faydalı yotum derecesi. Örn:. 2/3
#reviewText – Kullanıcın yazdığı inceleme metni
#overall – Ürün rating’i
#summary - İnceleme özeti
#unixReviewTime – İnceleme zamanı (unix time)
#reviewTime – İnceleme zamanı (raw)

# credentials.
creds = {'user': 'group4',
         'passwd': 'haydegidelum',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'group4'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'


# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

query = "select * from df_sub.csv limit 5"
pd.read_sql_query(query, conn)


query = "select asin, count(asin) as count from reviews" \
        " group by asin order by count desc limit 10"
data_count_asin = pd.read_sql_query(query, conn)
data_count_asin.head()


###################################################
# Adım 3. En fazla yorum alan ürüne göre veri setini indirgeyiniz (df_sub)
###################################################
pd.set_option('display.max_columns', None)
data = pd.read_csv(r"C:\Users\Suleakcay\PycharmProjects\pythonProject3\Datasets\df_sub.csv")
df_sub = data.copy()
df_sub.head()
df_sub.info()
df_sub.columns
df_sub.shape
df_sub.isnull().sum() #reviewrName(1) ve reviewText(1) eksik değer var
df_sub.head()
###################################################
# Adım 4. Ürünün ortalama puanı nedir?
###################################################
df_sub["mean"] = df_sub["overall"].mean()
#4.587589013224822


###################################################
# Adım 5. Tarihe ağırlıklı puan ortalaması hesaplayınız.
###################################################

# day_diff hesaplamak için: (yorum sonrası ne kadar gün geçmiş)
df_sub['reviewTime'] = pd.to_datetime(df_sub['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df_sub["day_diff"] = (current_date - df_sub['reviewTime']).dt.days

# Zamanı çeyrek değerlere göre bölüyorum.
a = df_sub["day_diff"].quantile(0.25)
#281.0
b = df_sub["day_diff"].quantile(0.50)
#431.0
c = df_sub["day_diff"].quantile(0.75)
#601.0

###################################################
# Adım 6. Önceki maddeden gelen a,b,c değerlerine göre ağırlıklı puanı hesaplayınız.
###################################################
df_sub['weighted'] = df_sub.loc[(df_sub['day_diff'] <= a), 'overall'].mean() *27 /100 + \
    df_sub.loc[(df_sub['day_diff'] > a) & (df_sub['day_diff'] <= b), 'overall'].mean() *28 /100 + \
    df_sub.loc[(df_sub['day_diff'] > b) & (df_sub['day_diff'] <= c), 'overall'].mean() * 20 / 100 + \
    df_sub.loc[(df_sub['day_diff'] > c), 'overall'].mean() * 25 / 100

df_sub['weighted'].mean() #bir önceki ortalama 4.587589013224822  şimdiki 4.591879221719786


###################################################
# Görev 2: Product tanıtım sayfasında görüntülenecek ilk 20 yorumu belirleyiniz.
###################################################

###################################################
# Adım 1. Helpful değişkeni içerisinden 3 değişken türetiniz. 1: helpful_yes, 2: helpful_no,  3: total_vote
###################################################

# Helpful içerisinde 2 değer vardır. Birincisi yorumları faydalı bulan oy sayısı ikincisi toplam oy sayısı.
# Dolayısıyla önce ikisini ayrı ayrı çekmeli sonra da (total_vote - helpful_yes) yaparak helpful_no'yu hesaplamalısınız
#helpful(O)
dataframe = pd.DataFrame()
dataframe['helpful'] = [i for i in df_sub['helpful'].values]
df_sub['helpful'] = df_sub['helpful'].apply(lambda x: ast.literal_eval(x))
type(df_sub['helpful'][0])
type(ast.literal_eval(df_sub['helpful'][0]))

df_sub["helpful_yes"] = df_sub['helpful'].apply(lambda x: x[0])
df_sub["helpful_no"] = df_sub['helpful'].apply(lambda x: x[1])
df_sub["total_vote"] = df_sub["helpful_yes"] + df_sub["helpful_no"]
df_sub.head()


###################################################
# Adım 2. score_pos_neg_diff'a göre skorlar oluşturunuz ve df_sub içerisinde score_pos_neg_diff ismiyle kaydediniz.
###################################################
def score_pos_neg_diff(positive_score, negative_score):
    return positive_score - negative_score

df_sub["score_pos_neg_diff"] = df_sub.apply(lambda x: score_pos_neg_diff(x['helpful_yes'], x['helpful_no']),axis=1)
df_sub.sort_values('score_pos_neg_diff', ascending=False).head()

###################################################
# Adım 3. score_average_rating'a göre skorlar oluşturunuz ve df_sub içerisinde score_average_rating ismiyle kaydediniz.
###################################################
# Score = Average rating = (Positive ratings) / (Total ratings)

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)
df_sub["score_average_rating"] = df_sub.apply(lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']),axis=1)
df_sub.sort_values('score_average_rating', ascending=False).head()

##################################################
# Adım 4. wilson_lower_bound'a göre skorlar oluşturunuz ve df_sub içerisinde wilson_lower_bound ismiyle kaydediniz.
###################################################
def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df_sub["wilson_lower_bound"] = df_sub.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)
df_sub.sort_values('wilson_lower_bound', ascending=False).head()

##################################################
# Adım 5. Ürün sayfasında gösterilecek 20 yorumu belirleyiniz ve sonuçları yorumlayınız.
###################################################
df_sub.columns
df_sub[['summary', 'helpful', 'overall', 'mean', 'weighted', 'score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']]
df_sub.sort_values('wilson_lower_bound', ascending=False).head(20)


#ÇIKTISI:
#          reviewerID        asin                          reviewerName       helpful                                         reviewText  overall                                            summary  unixReviewTime reviewTime  mean  day_diff  weighted  helpful_yes  helpful_no  total_vote  score_pos_neg_diff  score_average_rating  wilson_lower_bound
#2031  A12B7ZMXFI6IXY  B007WTAJTO                  Hyoun Kim "Faluzure"  [1952, 2020]  [[ UPDATE - 6/19/2014 ]]So my lovely wife boug...     5.00  UPDATED - Great w/ Galaxy S4 & Galaxy Tab 4 10...      1367366400 2013-01-05  4.59       702      4.59         1952        2020        3972                 -68                  0.49                0.48
#3449   AOEAD7DPLZE53  B007WTAJTO                     NLee the Engineer  [1428, 1505]  I have tested dozens of SDHC and micro-SDHC ca...     5.00  Top of the class among all (budget-priced) mic...      1348617600 2012-09-26  4.59       803      4.59         1428        1505        2933                 -77                  0.49                0.47
#4212   AVBMZZAFEKO58  B007WTAJTO                           SkincareCEO  [1568, 1694]  NOTE:  please read the last update (scroll to ...     1.00  1 Star reviews - Micro SDXC card unmounts itse...      1375660800 2013-05-08  4.59       579      4.59         1568        1694        3262                -126                  0.48                0.46
#317   A1ZQAQFYSXL5MQ  B007WTAJTO               Amazon Customer "Kelly"    [422, 495]  If your card gets hot enough to be painful, it...     1.00                                Warning, read this!      1346544000 2012-02-09  4.59      1033      4.59          422         495         917                 -73                  0.46                0.43
#3981  A1K91XXQ6ZEBQR  B007WTAJTO            R. Sutton, Jr. "RWSynergy"    [112, 139]  The last few days I have been diligently shopp...     5.00  Resolving confusion between "Mobile Ultra" and...      1350864000 2012-10-22  4.59       777      4.59          112         139         251                 -27                  0.45                0.39
#1835  A1J6VSUM80UAF8  B007WTAJTO                           goconfigure      [60, 68]  Bought from BestBuy online the day it was anno...     5.00                                           I own it      1393545600 2014-02-28  4.59       283      4.59           60          68         128                  -8                  0.47                0.38
#4672  A2DKQQIZ793AV5  B007WTAJTO                               Twister      [45, 49]  Sandisk announcement of the first 128GB micro ...     5.00  Super high capacity!!!  Excellent price (on Am...      1394150400 2014-07-03  4.59       158      4.59           45          49          94                  -4                  0.48                0.38
#4596  A1WTQUOQ4WG9AI  B007WTAJTO           Tom Henriksen "Doggy Diner"     [82, 109]  Hi:I ordered two card and they arrived the nex...     1.00     Designed incompatibility/Don't support SanDisk      1348272000 2012-09-22  4.59       807      4.59           82         109         191                 -27                  0.43                0.36
#4306   AOHXKM5URSKAB  B007WTAJTO                         Stellar Eller      [51, 65]  While I got this card as a "deal of the day" o...     5.00                                      Awesome Card!      1339200000 2012-09-06  4.59       823      4.59           51          65         116                 -14                  0.44                0.35
#315   A2J26NNQX6WKAU  B007WTAJTO            Amazon Customer "johncrea"      [38, 48]  Bought this card to use with my Samsung Galaxy...     5.00  Samsung Galaxy Tab2 works with this card if re...      1344816000 2012-08-13  4.59       847      4.59           38          48          86                 -10                  0.44                0.34
#3807   AFGRMORWY2QNX  B007WTAJTO                            R. Heisler      [22, 25]  I bought this card to replace a lost 16 gig in...     3.00   Good buy for the money but wait, I had an issue!      1361923200 2013-02-27  4.59       649      4.59           22          25          47                  -3                  0.47                0.33
#4302  A2EL2GWJ9T0DWY  B007WTAJTO                             Stayeraug      [14, 16]  So I got this SD specifically for my GoPro Bla...     5.00                        Perfect with GoPro Black 3+      1395360000 2014-03-21  4.59       262      4.59           14          16          30                  -2                  0.47                0.30
#93     A837QPVOZ9YAD  B007WTAJTO                               Airedad      [15, 21]  I'm amazed.  I ordered this from Amazon on Tue...     5.00  Very fast class 10 card - and excellent servic...      1343174400 2012-07-25  4.59       866      4.59           15          21          36                  -6                  0.42                0.27
#1609  A2TPXOZSU1DACQ  B007WTAJTO                                Eskimo        [7, 7]  I have always been a sandisk guy.  This cards ...     5.00                  Bet you wish you had one of these      1395792000 2014-03-26  4.59       257      4.59            7           7          14                   0                  0.50                0.27
#1465   A6I8KXYK24RTB  B007WTAJTO                              D. Stein        [7, 7]  I for one have not bought into Google's, or an...     4.00                                           Finally.      1397433600 2014-04-14  4.59       238      4.59            7           7          14                   0                  0.50                0.27
#4072  A22GOZTFA02O2F  B007WTAJTO                           sb21 "sb21"        [6, 6]  I used this for my Samsung Galaxy Tab 2 7.0 . ...     5.00               Used for my Samsung Galaxy Tab 2 7.0      1347321600 2012-11-09  4.59       759      4.59            6           6          12                   0                  0.50                0.25
#2268   A680RUE1FDO8B  B007WTAJTO                      Jerry Saperstein       [8, 10]  My Samsung Galaxy S4 now has 119GB of fast mic...     5.00                      Incredible simply incredible.      1395014400 2014-03-17  4.59       266      4.59            8          10          18                  -2                  0.44                0.25
#2583  A3MEPYZVTAV90W  B007WTAJTO                               J. Wong        [5, 5]  I bought this Class 10 SD card for my GoPro 3 ...     5.00                  Works Great with a GoPro 3 Black!      1370649600 2013-08-06  4.59       489      4.59            5           5          10                   0                  0.50                0.24
#1142  A1PLHPPAJ5MUXG  B007WTAJTO  Daniel Pham(Danpham_X @ yahoo.  com)        [5, 5]  As soon as I saw that this card was announced ...     5.00                          Great large capacity card      1396396800 2014-02-04  4.59       307      4.59            5           5          10                   0                  0.50                0.24
#1072  A2O96COBMVY9C4  B007WTAJTO                        Crysis Complex        [5, 5]  What more can I say? The 64GB micro SD works f...     5.00               Works wonders for the Galaxy Note 2!      1349395200 2012-05-10  4.59       942      4.59            5           5          10                   0                  0.50                0.24

#veri setinde yorumları help_yes ve help_no değerlerine göre wilson_lower_bound fonksiyonuna göre sıraladım.
#2268 indeksteki eleman bana kalırsa sıralamada daha aşağıda olabilirdi yes_help ve no_help poranına göre.
#Genel ortalamaları eşittir.
#2582 indeksteki yorum diğer 1142 ve 1072 indeksteki yorumlardaki score_average_rating değerlerinden daha düşüktür
#score_average_rating fonksiyonun doğruluğuna tam net güvenseydik 2582.indeksteki yorum sıralamada daha aşaağıda kalacaktı.
#Burada açıkça bizi kurtaran Wilson_lower_bound fonksiyonu olduğudur.Bellli bir güven aralığına göre hesaplanıyor.
#Buda 2582.indeksteki yorumun daha yukarıda olmasını sağlar...