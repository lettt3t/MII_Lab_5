from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn .linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd               
import matplotlib.pyplot as plt
import seaborn as sns
sns.set ()
import warnings
warnings.filterwarnings('ignore')



model_preds = []
list_r2_score = list()

def graph_score_r2():
    x = ['Linear_Regression', 'K-Neigbors Regressor','Random Forest']
    y = [list_r2_score[0],list_r2_score[1],list_r2_score[2]]
    plt.bar(x, y)
    plt.ylabel('Score R2')
    plt.title('R2 Score graph')
    plt.show()

def fit_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    list_r2_score.append(r2)
    mse = mean_squared_error(y_test, y_pred)
    model_preds.append([model_name, r2, mse])
    print("The Mean Squared error is: ", mse)


def model_eval():
    preds = pd.DataFrame(model_preds)
    preds.columns = ["Model Name", "R2 Score", "Mean Squared Error"]
    return preds.sort_values(by="R2 Score", ascending=False)


df = pd.read_csv ('fipe_2022.csv')
df.head()
df.info()

df.describe(include='all').round(0)


#cat_df = df[['month_of_reference', 'brand', 'model', 'fuel', 'gear']]
#num_df = df[['engine_size', 'year_model', 'avg_price_brl', 'age_years']]
le = LabelEncoder ()


cat_df = df.select_dtypes(exclude=["int", "float"])

for i in cat_df:
    cat_df[i] = le.fit_transform(df[i])


num_df = df.select_dtypes(include=['int', 'float'])
main_df = pd.concat([num_df, cat_df], axis=1)


X = main_df.drop(columns=["avg_price_brl"])
y = main_df["avg_price_brl"]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, random_state=0)

## Linear Regression
print("=========================")
print("Linear_Regression")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
fit_model(lr_model, "Linear Regression")
print("=========================")
## K-Neigbors Regressor
print("=========================")
print("K-Neigbors Regressor")
knn_model = KNeighborsRegressor(n_neighbors=6)
fit_model(knn_model, "K-Neigbors Regressor")
print("=========================")
## Random Forest
print("=========================")
print("Random Forest")
randfor_model = RandomForestRegressor()
fit_model(randfor_model, "Random Forest Regressor")
print("=========================")

graph_score_r2()
print(model_eval())
