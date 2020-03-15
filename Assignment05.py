import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from statistics import mean


def openData():
    df = pd.read_csv('data/ASS05_Data.csv')
    return df[['LotArea', 'TotalBsmtSF', 'GarageCars', 'AGE', 'TotalArea', 'SalePrice']]


def adjusted_r2(r2, n, k):
    """
    Calculated an adjusted R^2 value

    :param r2: R^2 value
    :param n: Sample size
    :param k:
    :return:
    """
    return 1 - (((1-r2)*(n-1))/(n-k-1))


def questionOneStep01():
    df = openData().rename(columns={"SalePrice": "TrueTarget"})
    return train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3)


def questionOneStep02():
    x_train, x_test, y_train, y_test = questionOneStep01()
    regModel = LinearRegression().fit(x_train, y_train)
    y_pred = regModel.predict(x_train)
    rSquared = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    adjRSquared = adjusted_r2(rSquared, x_train.shape[0], x_train.shape[1])
    print("Training R^2: ", rSquared)
    print("Training MSE: ", mse)
    print("Training Adjusted R^2: ", adjRSquared)
    return regModel, x_test, y_test


def questionOneStep03():
    regModel, x_test, y_test = questionOneStep02()
    y_pred = regModel.predict(x_test)
    score = r2_score(y_test, y_pred)
    print("Validation R^2 score: ", score)
    return y_test, y_pred


def questionOneStep04():
    y_test, y_pred = questionOneStep03()
    ase = mean_squared_error(y_test, y_pred)
    print("Validation ASE: ", ase)


def questionOneStep05():
    i = 0
    while i < 5:
        print("--------------Iteration", i + 1, "-----------------")
        questionOneStep04()
        i += 1


def questionTwoStep01And02():
    kf = KFold(shuffle=True)
    return kf.split(openData())


def questionTwoStep03Through06():
    df = openData()
    i = 1
    trainingMSE = []
    validationASE = []
    for train_index, test_index in questionTwoStep01And02():
        x_train = df.iloc[train_index].iloc[:, :-1]
        y_train = df.iloc[train_index].iloc[:, -1]
        x_test = df.iloc[test_index].iloc[:, :-1]
        y_test = df.iloc[test_index].iloc[:, -1]

        regModel = LinearRegression().fit(x_train, y_train)
        y_pred_train = regModel.predict(x_train)
        score_train = r2_score(y_train, y_pred_train)
        mse = mean_squared_error(y_train, y_pred_train)
        trainingMSE.append(mse)
        print("Training R^2 Score for partition", i, ":", score_train)
        print("Training MSE for partition", i, ":", mse)

        y_pred_test = regModel.predict(x_test)
        ase = mean_squared_error(y_test, y_pred_test)
        validationASE.append(ase)
        print("ASE for partition", i, ":", ase)
        i += 1
    print("Mean training ASE:", mean(trainingMSE))
    print("Mean Validation ASE:", mean(validationASE))


print("\n--------Problem 1-------\n")
questionOneStep05()
print("\n------------Problem 2--------\n")
questionTwoStep03Through06()

