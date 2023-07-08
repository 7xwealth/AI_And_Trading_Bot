import sys

import numpy as np
import pandas as pd
import yfinance
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from tensorflow.python.keras.losses import mean_squared_error

price_received = False


def main(symbol_):
    timeframe = "1Day"
    end = "2023-06-06"

    if symbol_ == "GOLD":
        start = "2000-01-01"
    if symbol_ == "TSLA":
        start = "2017-02-02"
    if symbol_ == "NVO":
        start = "2006-04-04"
    if symbol_ == "NVDA":
        start = "2015-06-06"
    if symbol_ == "AAPL":
        start = "2007-08-08"

    tcker_yahoo = yfinance.Ticker(symbol_)
    yahoo_data_df = tcker_yahoo.history(start=start, end=end)
    data_today = yahoo_data_df.iloc[-1]
    print("TODAY", data_today)

    # alpaca = api.REST("PKD1X62GUMH444FLA52Z", "1B9ku9WKoTFunlgftTUlGAWPx5lf48QFgbZN7KEd")
    # df = alpaca.get_bars(symbol_, timeframe, start, end).df

    # df = pd.read_csv(f"data/{symbol_}.us.csv")

    # yahoo_df_indc = RSIIndicator(yahoo_data_df["Close"])
    # yahoo_data_df["RSI"] = pd.Series(yahoo_df_indc.rsi())
    # yahoo_df_adx = ADXIndicator(yahoo_data_df["High"], yahoo_data_df["Low"], yahoo_data_df["Close"])
    # yahoo_data_df["ADX"] = pd.Series(yahoo_df_adx.adx())
    # yahoo_df_adx = MACD(yahoo_data_df["Close"])
    # yahoo_data_df["MACD"] = pd.Series(yahoo_df_adx.macd())
    # yahoo_df_adx = StochasticOscillator(yahoo_data_df["High"], yahoo_data_df["Low"], yahoo_data_df["Close"])
    # yahoo_data_df["STOCH"] = pd.Series(yahoo_df_adx.stoch())
    # yahoo_df_adx = EMAIndicator(yahoo_data_df["Close"])
    # yahoo_data_df["EMA"] = pd.Series(yahoo_df_adx.ema_indicator())

    yahoo_data_df = yahoo_data_df.fillna(value=yahoo_data_df.median())

    X_train, x_test = train_test_split(yahoo_data_df, test_size=0.2)
    y_train, y_test = X_train["Close"], x_test["Close"]
    X_train, x_test = X_train.drop(columns=["Close"]), x_test.drop(columns=["Close"])

    x_test = x_test[:1]
    x_test["Open"] = float(data_today["Open"])
    x_test["High"] = float(data_today["High"])
    x_test["Low"] = float(data_today["Low"])
    x_test["Volume"] = float(data_today["Volume"])
    close = float(data_today["Close"])

    min_max = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = min_max.fit_transform(X_train)
    x_test_scaled = min_max.transform(x_test)

    # for i in range(60, len(X_train_scaled)):
    #     X_train.append(pd.Series(X_train_scaled[i - 60:i, 0], name="Forecast"))

    X_train = np.array(X_train_scaled)

    print(x_test)
    print(x_test.shape)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    x_test = np.reshape(x_test_scaled, (x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))

    # if os.path.isfile(f"{symbol_}__MODEL.h5"):
    #     model = load_model(f"{symbol_}__MODEL.h5")
    # else:
    model = Sequential()

    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss=mean_squared_error)
    model.fit(X_train, y_train, epochs=70, batch_size=32)
    model.save(f"{symbol_}__MODEL.h5")

    predict = model.predict(x_test)
    print("pred : ", predict)
    print("real : ", y_test)

    client = TradingClient("YOUR", "KEYS") # AI KEY

    predicted = model.predict(x_test)

    # cross_val_score(model, X_train, y, cv=kfold, scoring='neg_mean_squared_error')
    # print(cross_val_score(model, x_test, y_test))
    buy_sell = [OrderSide.BUY, OrderSide.SELL]

    if close >= predicted[0]:
        long_short = 1
    else:
        long_short = 0

    print(f"Going {buy_sell[long_short]} on {symbol_} with price of ${close}")
    print(f"Predicted price is : {predicted}")

    market_order_data = MarketOrderRequest(
        symbol=symbol_,
        qty=(1900 // close),
        side=buy_sell[long_short],
        time_in_force=TimeInForce.GTC
    )
    client.submit_order(market_order_data)
    if symbol_ == "GOLD":
        sys.exit(0)


def blabla():
    client = TradingClient("YOUR", "KEYS")
    market_order_data = MarketOrderRequest(
        symbol="AAPL",
        qty=56,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    client.submit_order(market_order_data)


# tcker_yahoo = yfinance.Ticker("AAPL")
# lst = tcker_yahoo.history(interval="1m", period="1day")["Close"]
list__NVO, list__AAPL, list__TSLA, list__GOLD, list__NVDA = [], [], [], [], []
c__NVO, c__AAPL, c__TSLA, c__GOLD, c__NVDA = 0, 0, 0, 0, 0


def trading_bot_order(symbol_):
    timeframe = "1Day"
    end = "2023-06-06"

    if symbol_ == "GOLD":
        start = "2000-01-01"
    if symbol_ == "TSLA":
        start = "2017-02-02"
    if symbol_ == "NVO":
        start = "2006-04-04"
    if symbol_ == "NVDA":
        start = "2015-06-06"
    if symbol_ == "AAPL":
        start = "2007-08-08"

    async def bar_callback__AAPL(bar):
        client = TradingClient("YR", "KEYS")
        global list__AAPL, c__AAPL
        for property_name, value in bar:
            print(property_name, value, "c : ", c__AAPL)
            if property_name == "close":
                c__AAPL += 1
                list__AAPL.append(value)
                print("list AAPL : ", list__AAPL)
                print("At run through number : ", c__AAPL, " Close value => ", value)
                rsi_indecator = RSIIndicator(pd.Series(list__AAPL))
                print("current rsi : ", rsi_indecator.rsi())
                if len(rsi_indecator.rsi()) >= 2:
                    if rsi_indecator.rsi().iloc[-1] <= 32 and rsi_indecator.rsi().iloc[-2] >= 32 and c__AAPL >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="AAPL",
                            qty=(2000 // value),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed BUY order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1], "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])
                    elif rsi_indecator.rsi().iloc[-1] > 70 and rsi_indecator.rsi().iloc[-2] <= 70 and c__AAPL >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="AAPL",
                            qty=(2000 // value),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed SELL order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1],
                              "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])

    async def bar_callback__TSLA(bar):
        client = TradingClient("YR", "KEYS")
        global list__TSLA, c__TSLA
        for property_name, value in bar:
            print(property_name, value, "c : ", c__TSLA)
            if property_name == "close":
                c__TSLA += 1
                list__TSLA.append(value)
                print("list TSLA : ", list__TSLA)
                print("At run through number : ", c__TSLA, " Close value => ", value)
                rsi_indecator = RSIIndicator(pd.Series(list__TSLA))
                print("current rsi : ", rsi_indecator.rsi())
                if len(rsi_indecator.rsi()) >= 2:
                    if rsi_indecator.rsi().iloc[-1] <= 32 and rsi_indecator.rsi().iloc[-2] >= rsi_indecator.rsi().iloc[
                        -1] and c__TSLA >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="TSLA",
                            qty=(2000 // value),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed BUY order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1], "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])
                    elif rsi_indecator.rsi().iloc[-1] > 70 and rsi_indecator.rsi().iloc[-2] <= rsi_indecator.rsi().iloc[
                        -1] and c__TSLA >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="TSLA",
                            qty=(2000 // value),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed SELL order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1],
                              "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])

    async def bar_callback__NVO(bar):
        client = TradingClient("YR", "KEYS")
        global list__NVO, c__NVO
        for property_name, value in bar:
            print(property_name, value, "c : ", c__NVO)
            if property_name == "close":
                c__NVO += 1
                list__NVO.append(value)
                print("list NVO : ", list__NVO)
                print("At run through number : ", c__NVO, " Close value => ", value)
                rsi_indecator = RSIIndicator(pd.Series(list__NVO))
                print("current rsi : ", rsi_indecator.rsi())
                if len(rsi_indecator.rsi()) >= 2:
                    if rsi_indecator.rsi().iloc[-1] <= 32 and rsi_indecator.rsi().iloc[-2] >= 32 and c__NVO >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="NVO",
                            qty=(2000 // value),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed BUY order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1], "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])
                    elif rsi_indecator.rsi().iloc[-1] > 70 and rsi_indecator.rsi().iloc[-2] <= 70 and c__NVO >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="NVO",
                            qty=(2000 // value),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed SELL order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1],
                              "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])

    async def bar_callback__NVDA(bar):
        client = TradingClient("YR", "KEYS")
        global list__NVDA, c__NVDA
        for property_name, value in bar:
            print(property_name, value, "c : ", c__NVDA)
            if property_name == "close":
                c__NVDA += 1
                list__NVDA.append(value)
                print("list NVDA : ", list__NVDA)
                print("At run through number : ", c__NVDA, " Close value => ", value)
                rsi_indecator = RSIIndicator(pd.Series(list__NVDA))
                print("current rsi : ", rsi_indecator.rsi())
                if len(rsi_indecator.rsi()) >= 2:
                    if rsi_indecator.rsi().iloc[-1] <= 32 and rsi_indecator.rsi().iloc[-2] >= 32 and c__NVDA >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="NVDA",
                            qty=(2000 // value),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed BUY order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1], "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])
                    elif rsi_indecator.rsi().iloc[-1] > 70 and rsi_indecator.rsi().iloc[-2] <= 70 and c__NVDA >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="NVDA",
                            qty=(2000 // value),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed SELL order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1],
                              "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])

    async def bar_callback__GOLD(bar):
        client = TradingClient("YOUR", "KEYS")
        global list__GOLD, c__GOLD
        for property_name, value in bar:
            print(property_name, value, "c : ", c__GOLD)
            if property_name == "close":
                c__GOLD += 1
                list__GOLD.append(value)
                print("list GOLD : ", list__GOLD)
                print("At run through number : ", c__GOLD, " Close value => ", value)
                rsi_indecator = RSIIndicator(pd.Series(list__GOLD))
                print("current rsi : ", rsi_indecator.rsi())
                if len(rsi_indecator.rsi()) >= 2:
                    if rsi_indecator.rsi().iloc[-1] <= 32 and rsi_indecator.rsi().iloc[-2] >= 32 and c__GOLD >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="GOLD",
                            qty=(2000 // value),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed BUY order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1], "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])
                    elif rsi_indecator.rsi().iloc[-1] > 70 and rsi_indecator.rsi().iloc[-2] <= 70 and c__GOLD >= 13:
                        market_order_data = MarketOrderRequest(
                            symbol="GOLD",
                            qty=(2000 // value),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        client.submit_order(market_order_data)
                        print("LONG : Executed SELL order, at RSI of -1 of ", rsi_indecator.rsi().iloc[-1],
                              "and -2 of ",
                              rsi_indecator.rsi().iloc[-2])

    # rsi = RSIIndicator(close=yahoo_data_df["Close"])
    data_stream = StockDataStream("YOUR", "APIKEYS")
    data_stream.subscribe_bars(bar_callback__AAPL, "AAPL")
    data_stream.subscribe_bars(bar_callback__TSLA, "TSLA")
    data_stream.subscribe_bars(bar_callback__NVO, "NVO")
    data_stream.subscribe_bars(bar_callback__NVDA, "NVDA")
    data_stream.subscribe_bars(bar_callback__GOLD, "GOLD")
    data_stream.run()


if __name__ == '__main__':
    symbols = ["AAPL", "TSLA", "NVO", "NVDA", "GOLD"]
    for symbol in symbols:
        main(symbol)
