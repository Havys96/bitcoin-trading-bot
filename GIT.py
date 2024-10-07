import logging
import time
import math
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import os
from dotenv import load_dotenv
import sys
import signal
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import traceback

# .env 파일 로드
load_dotenv()

# 상수 정의
SIDE_BUY = 'BUY'
SIDE_SELL = 'SELL'
ORDER_TYPE_STOP_MARKET = 'STOP_MARKET'
ORDER_TYPE_TAKE_PROFIT_MARKET = 'TAKE_PROFIT_MARKET'
ORDER_TYPE_TRAILING_STOP_MARKET = 'TRAILING_STOP_MARKET'
ORDER_TYPE_MARKET = 'MARKET'

# 바이낸스 메인넷 API 설정 (환경 변수 사용)
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    raise ValueError("환경 변수 BINANCE_API_KEY 및 BINANCE_API_SECRET을 설정해야 합니다.")

client = Client(api_key, api_secret, testnet=False)  # 실제 거래 시 testnet=False

# 로깅 설정 (파일과 콘솔에 동시에 로그 기록)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 파일 핸들러 설정
file_handler = logging.FileHandler('../../../my-project/trading_bot.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 전역 변수로 누적 손익 추적
total_profit_loss = 0  # 누적 수익/손실
total_profit = 0        # 누적 익절 수익
total_loss = 0          # 누적 손절 손실

# 자본금 및 투자 현황 변수 설정
initial_investment = 152  # 총 자본금 설정 (예시로 $152)
current_investment_percentage = 0  # 현재 사용 중인 자본금 퍼센트
max_investment_percentage = 100  # 최대 사용 가능한 자본금 비율 (100%)
investment_step_percentage = 10  # 각 단계마다 사용하는 자본금 비율 (10%)
entry_count = 0  # 현재 진입 횟수
max_entry_count = 10  # 최대 진입 횟수 설정

# 종료 신호 핸들러
def signal_handler(sig, frame):
    logging.info("스크립트를 수동으로 종료했습니다.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 리스크 관리 함수 (예: 최대 손실 한도 설정)
def risk_management(balance):
    global total_profit_loss
    max_drawdown = 0.4  # 예: 최대 40% 손실 허용
    if total_profit_loss <= -initial_investment * max_drawdown:
        logging.critical("최대 손실 한도를 초과했습니다. 봇을 종료합니다.")
        sys.exit(1)

# 레버리지 설정 함수
def set_leverage(symbol, leverage):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logging.info(f"{symbol}에 레버리지 {leverage}배로 설정 완료!")
    except BinanceAPIException as e:
        logging.error(f"레버리지 설정 API 에러: {e}")
    except BinanceOrderException as e:
        logging.error(f"레버리지 설정 주문 에러: {e}")
    except Exception as e:
        logging.error(f"레버리지 설정 에러: {e}")

# 현재 레버리지 확인 함수
def get_current_leverage(symbol):
    try:
        # 레버리지 정보 조회
        account_info = client.futures_account()
        positions = account_info['positions']
        for pos in positions:
            if pos['symbol'] == symbol:
                leverage = pos.get('leverage')
                if leverage:
                    logging.info(f"{symbol}의 현재 레버리지: {leverage}배")
                    return int(leverage)
        logging.error(f"{symbol}에 대한 레버리지 정보를 찾을 수 없습니다.")
        return None
    except BinanceAPIException as e:
        logging.error(f"레버리지 조회 API 에러: {e}")
    except Exception as e:
        logging.error(f"레버리지 조회 오류: {e}")
    return None

# 심볼의 필터 정보 가져오기 (재시도 로직 추가)
def get_symbol_info(symbol, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            exchange_info = client.futures_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    logging.info(f"{symbol}의 심볼 정보 가져오기 성공.")
                    return s
            logging.error(f"심볼 정보 없음: {symbol}")
            return None
        except BinanceAPIException as e:
            logging.error(f"심볼 정보 조회 API 에러: {e}, {retries + 1}번째 시도 중")
            retries += 1
            time.sleep(2)  # 잠시 대기 후 재시도
        except Exception as e:
            logging.error(f"심볼 정보 조회 오류: {e}, {retries + 1}번째 시도 중")
            retries += 1
            time.sleep(2)  # 잠시 대기 후 재시도
    logging.error(f"심볼 정보 조회에 실패했습니다. 최대 재시도 횟수를 초과했습니다.")
    return None

# 소수점 자릿수 맞추기 함수
def format_quantity(symbol, quantity):
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        return None
    for f in symbol_info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            precision = int(-math.log10(step_size))
            return round(quantity, precision)
    return quantity

def format_price(symbol, price):
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        return None
    for f in symbol_info['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            tick_size = float(f['tickSize'])
            precision = int(-math.log10(tick_size))
            return round(price, precision)
    return price

# 잔고 확인 함수 (USDT 사용 가능 잔고 확인)
def get_available_balance():
    try:
        balance_info = client.futures_account_balance()
        for asset in balance_info:
            if asset['asset'] == 'USDT':
                available_balance = float(asset['availableBalance'])
                logging.info(f"현재 사용 가능 USDT 잔고: {available_balance} USDT")
                return available_balance
    except BinanceAPIException as e:
        logging.error(f"잔고 조회 API 에러: {e}")
    except Exception as e:
        logging.error(f"잔고 조회 오류: {e}")
    return None

# 현재 가격 조회 함수
def get_current_price(symbol):
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        logging.info(f"현재 {symbol} 가격: {price} USDT")
        return price
    except BinanceAPIException as e:
        logging.error(f"현재 가격 조회 API 에러: {e}")
    except Exception as e:
        logging.error(f"현재 가격 조회 오류: {e}")
    return None

# 최소 주문 수량 확인 함수 (필터 정보가 없을 경우 기본값 설정)
def get_min_order_quantity(symbol):
    try:
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"{symbol}의 필수 필터 정보가 누락되었습니다. 기본값으로 진행합니다.")
            return 0.001, 0.001, 5  # 기본값 설정: 최소 수량, 스텝 사이즈, 최소 명목가
        min_qty = None
        step_size = None
        min_notional = None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])
                min_qty = float(f['minQty'])
            elif f['filterType'] == 'MIN_NOTIONAL':
                min_notional = float(f.get('notional', 5))
        if min_qty and step_size and min_notional:
            logging.info(f"{symbol}의 최소 주문 수량: {min_qty}, 스텝 사이즈: {step_size}, 최소 명목가: {min_notional} USDT")
            return min_qty, step_size, min_notional
        else:
            logging.error(f"{symbol}의 필수 필터 정보가 누락되었습니다. 기본값으로 진행합니다.")
            return 0.001, 0.001, 5  # 기본값 설정
    except BinanceAPIException as e:
        logging.error(f"심볼 정보 조회 API 에러: {e}")
    except Exception as e:
        logging.error(f"심볼 정보 조회 오류: {e}")
    return 0.001, 0.001, 5  # 기본값 설정

# 모든 오픈 포지션 조회 함수
def get_open_positions():
    try:
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        if open_positions:
            for pos in open_positions:
                logging.info(f"열려 있는 포지션: 심볼={pos['symbol']}, 수량={pos['positionAmt']}, 진입 가격={pos['entryPrice']}")
        else:
            logging.info("열려 있는 포지션이 없습니다.")
        return open_positions
    except BinanceAPIException as e:
        logging.error(f"포지션 조회 API 에러: {e}")
    except Exception as e:
        logging.error(f"포지션 조회 오류: {e}")
    return []

# 모든 오픈 포지션 청산 함수
def close_all_positions():
    global current_investment_percentage, entry_count, total_profit_loss, total_profit, total_loss
    open_positions = get_open_positions()
    for pos in open_positions:
        symbol = pos['symbol']
        side = SIDE_SELL if float(pos['positionAmt']) > 0 else SIDE_BUY
        quantity = abs(float(pos['positionAmt']))
        try:
            # 현재 미실현 손익 계산
            if 'unrealizedProfit' in pos:
                unrealized_pnl = float(pos['unrealizedProfit'])
            elif 'unRealizedProfit' in pos:
                unrealized_pnl = float(pos['unRealizedProfit'])
            else:
                logging.error("'unrealizedProfit' 키를 찾을 수 없습니다.")
                unrealized_pnl = 0.0

            # 수수료 계산 (예: 거래 금액의 0.04% * 2)
            fees = quantity * float(pos['entryPrice']) * 0.0004 * 2
            net_pnl = unrealized_pnl - fees
            total_profit_loss += net_pnl
            if net_pnl > 0:
                total_profit += net_pnl
            else:
                total_loss += net_pnl
            logging.info(f"포지션 청산 전 손익: {unrealized_pnl} USDT, 수수료: {fees} USDT, 순손익: {net_pnl} USDT")

            logging.info(f"포지션 청산 시도 - 심볼: {symbol}, 방향: {side}, 수량: {quantity} BTC")
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=format_quantity(symbol, quantity)
            )
            logging.info(f"포지션 청산 완료: 심볼={symbol}, 방향={side}, 수량={quantity} BTC")
            # 포지션 청산 후 해당 심볼의 스탑 주문 취소
            cancel_stop_orders(symbol)
        except BinanceAPIException as e:
            logging.error(f"{side} 포지션 청산 API 에러: {e}")
        except BinanceOrderException as e:
            logging.error(f"{side} 포지션 청산 주문 에러: {e}")
        except Exception as e:
            logging.error(f"{side} 포지션 청산 에러: {e}")
    # 포지션 청산 후 투자 현황 초기화
    current_investment_percentage = 0
    entry_count = 0
    logging.info("모든 포지션이 청산되었습니다. 투자 현황이 초기화됩니다.")
    logging.info(f"누적 수익/손실: {total_profit_loss} USDT, 누적 수익: {total_profit} USDT, 누적 손실: {total_loss} USDT")
    # 새로운 매매 신호를 기다리기 위해 잠시 대기
    time.sleep(180)


# 활성화된 손절/익절 주문 조회 함수
def get_active_stop_orders(symbol):
    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        stop_orders = [order for order in orders if
                      order['type'] in [ORDER_TYPE_STOP_MARKET, ORDER_TYPE_TAKE_PROFIT_MARKET, ORDER_TYPE_TRAILING_STOP_MARKET]]
        logging.info(f"활성화된 손절/익절 주문 수: {len(stop_orders)}")
        return stop_orders
    except BinanceAPIException as e:
        logging.error(f"활성화된 손절/익절 주문 조회 API 에러: {e}")
        return []
    except Exception as e:
        logging.error(f"활성화된 손절/익절 주문 조회 오류: {e}")
        return []

# 활성화된 손절/익절 주문 취소 함수
def cancel_stop_orders(symbol):
    stop_orders = get_active_stop_orders(symbol)
    for order in stop_orders:
        try:
            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
            logging.info(f"취소된 손절/익절 주문: ID {order['orderId']}, 타입 {order['type']}")
        except BinanceAPIException as e:
            logging.error(f"손절/익절 주문 취소 API 에러: {e}")
        except Exception as e:
            logging.error(f"손절/익절 주문 취소 오류: {e}")

# 동적 투자 비율 계산 함수
def calculate_dynamic_investment_percentage(rsi, ema20, ema60):
    investment_percentage = investment_step_percentage  # 기본 설정된 투자 비율 사용
    # RSI에 따른 추가/감소 투자 비율
    if rsi < 20:
        investment_percentage += 5
    elif rsi < 30:
        investment_percentage += 3
    elif rsi > 80:
        investment_percentage -= 5
    elif rsi > 70:
        investment_percentage -= 3
    # EMA 교차 상태에 따른 조정
    if ema20 > ema60:
        investment_percentage += 2
    elif ema20 < ema60:
        investment_percentage -= 2
    # 투자 비율 범위 제한 (최소 1%, 최대 20%)
    investment_percentage = max(1, min(investment_percentage, 20))
    return investment_percentage

# 매매 주문 함수 (시장가 주문)
def place_order(symbol, side, quantity, reason):
    global current_investment_percentage, entry_count
    try:
        # 기존의 스탑 주문 취소
        cancel_stop_orders(symbol)

        # 현재 사용 가능 잔고 확인
        available_balance = get_available_balance()
        current_price = get_current_price(symbol)
        if current_price is None:
            logging.error("현재 가격을 가져올 수 없어 주문을 진행할 수 없습니다.")
            return None
        required_margin = quantity * current_price / 10  # 레버리지 10배 사용 시 요구 마진 계산
        logging.info(f"주문 시도 전 사용 가능 잔고: {available_balance} USDT, 요구 마진: {required_margin} USDT")
        if available_balance < required_margin:
            logging.error("사용 가능 잔고가 충분하지 않아 포지션을 열 수 없습니다.")
            return None

        # 자본금 비율 확인
        if current_investment_percentage + investment_step_percentage > max_investment_percentage:
            logging.error("현재 자본금 사용률이 최대 허용치를 초과하여 포지션을 열 수 없습니다.")
            return None

        logging.info(f"매매 주문 시도 - {side}, 수량: {quantity} BTC, 이유: {reason}")
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=format_quantity(symbol, quantity)
        )
        logging.info(f"포지션 진입 완료: {side}, 수량: {quantity} BTC")
        current_investment_percentage += investment_step_percentage  # 자본금 단계적 사용
        entry_count += 1  # 진입 횟수 증가

        # 투자 현황 로깅
        remaining_percentage = 100 - current_investment_percentage
        remaining_capital = initial_investment * remaining_percentage / 100
        logging.info(f"현재 진입 횟수: {entry_count}, 사용된 자본금: {current_investment_percentage}% ({initial_investment * current_investment_percentage / 100} USDT), 남은 자본금: {remaining_percentage}% ({remaining_capital} USDT)")

        return order
    except BinanceAPIException as e:
        logging.error(f"{side} 포지션 진입 API 에러: {e}")
    except BinanceOrderException as e:
        logging.error(f"{side} 포지션 진입 주문 에러: {e}")
    except Exception as e:
        logging.error(f"{side} 포지션 진입 에러: {e}")
    return None

# 손절/익절 설정 및 수익률 계산 함수
def set_stop_loss_and_take_profit(symbol, order, stop_loss_percentage=0.4, take_profit_percentage=0.8, trailing_stop_percentage=0.3, fees_percentage=0.8):
    try:
        if order is None:
            logging.warning("유효한 주문이 없어 손절/익절을 설정할 수 없습니다.")
            return

        # 손절매 주문 설정을 제외하여 손절매에 도달했을 때 포지션을 청산하지 않도록 함
        # 익절 주문 및 트레일링 스탑은 설정

        # 진입 가격 가져오기
        entry_price = None
        positions = get_open_positions()  # 오픈 포지션에서 진입 가격 가져오기
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                entry_price = float(pos['entryPrice'])
                break

        if entry_price is None or entry_price == 0:
            logging.error("진입 가격을 가져올 수 없어 손절/익절을 설정할 수 없습니다.")
            return

        # 수량 가져오기
        quantity = float(order['origQty']) if 'origQty' in order and order['origQty'] else 0
        if quantity == 0:
            positions = get_open_positions()  # 포지션 정보에서 수량 가져오기
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                    quantity = abs(float(pos['positionAmt']))
                    break

        if quantity == 0:
            logging.error("수량을 가져올 수 없어 손절/익절을 설정할 수 없습니다.")
            return

        logging.info(f"진입 가격: {entry_price} USDT, 수량: {quantity} BTC")

        # 포지션 방향에 따른 익절가 계산 (수수료 포함)
        position_side = order['side']  # 'BUY' 또는 'SELL'
        adjusted_take_profit_percentage = take_profit_percentage + fees_percentage

        if position_side == SIDE_BUY:
            take_profit_price = entry_price * (1 + adjusted_take_profit_percentage / 100)
        elif position_side == SIDE_SELL:
            take_profit_price = entry_price * (1 - adjusted_take_profit_percentage / 100)
        else:
            logging.error(f"알 수 없는 포지션 방향: {position_side}")
            return

        if take_profit_price <= 0:
            logging.error("익절가 계산 오류")
            return

        logging.info(f"익절가: {take_profit_price} USDT (수수료 포함)")

        # 익절 주문 설정
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if position_side == SIDE_BUY else SIDE_BUY,
            type=ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=format_price(symbol, take_profit_price),
            quantity=quantity
        )
        logging.info(f"익절 주문 설정 완료: 가격 {take_profit_price} USDT")

        # 트레일링 스탑 주문 설정
        trailing_stop_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if position_side == SIDE_BUY else SIDE_BUY,
            type=ORDER_TYPE_TRAILING_STOP_MARKET,
            quantity=quantity,
            callbackRate=trailing_stop_percentage  # 트레일링 스탑 비율 설정 (예: 0.3%)
        )
        logging.info(f"트레일링 스탑 주문 설정 완료: 트레일링 스탑 {trailing_stop_percentage}%")

    except BinanceAPIException as e:
        logging.error(f"손절/익절 주문 설정 API 에러: {e}")
    except BinanceOrderException as e:
        logging.error(f"손절/익절 주문 설정 주문 에러: {e}")
    except Exception as e:
        logging.error(f"손절/익절 주문 설정 오류: {e}")

# 기술적 지표 사용하여 추세 분석 함수 (RSI, EMA20, EMA60)
def get_technical_indicators(symbol, interval='5m', period=14, max_retries=5):
    retries = 0
    cached_data = None
    while retries < max_retries:
        try:
            if cached_data is None:
                klines = client.futures_klines(symbol=symbol, interval=interval, limit=200)
                close_prices = [float(kline[4]) for kline in klines if len(kline) > 4]
                cached_data = close_prices  # 데이터 캐싱
            else:
                close_prices = cached_data

            if len(close_prices) < max(20, 60, period):
                retries += 1
                cached_data = None
                time.sleep(10)
                continue

            data = pd.DataFrame({'close': close_prices})

            # RSI 계산
            rsi_indicator = RSIIndicator(close=data['close'], window=period)
            data['rsi'] = rsi_indicator.rsi()
            rsi = data['rsi'].iloc[-1]

            # EMA20 계산
            ema20_indicator = EMAIndicator(close=data['close'], window=20)
            data['ema20'] = ema20_indicator.ema_indicator()
            ema20 = data['ema20'].iloc[-1]

            # EMA60 계산
            ema60_indicator = EMAIndicator(close=data['close'], window=60)
            data['ema60'] = ema60_indicator.ema_indicator()
            ema60 = data['ema60'].iloc[-1]

            if pd.isna(rsi) or pd.isna(ema20) or pd.isna(ema60):
                retries += 1
                cached_data = None
                time.sleep(10)
                continue

            logging.info(f"현재 {symbol}의 RSI: {rsi}, EMA20: {ema20}, EMA60: {ema60}")
            return rsi, ema20, ema60

        except BinanceAPIException as e:
            logging.error(f"기술적 지표 조회 API 에러: {e}")
            retries += 1
            cached_data = None
            time.sleep(10)
        except KeyError as e:
            logging.error(f"KeyError 발생: {e}")
            retries += 1
            cached_data = None
            time.sleep(10)
        except Exception as e:
            logging.error(f"기술적 지표 조회 오류: {e}")
            retries += 1
            cached_data = None
            time.sleep(10)

    logging.error("기술적 지표 조회에 실패했습니다. 최대 재시도 횟수를 초과했습니다.")
    return None, None, None

# 포지션 손익 상태 가져오기 함수
def get_position_profit_loss(symbol):
    try:
        positions = client.futures_position_information(symbol=symbol)
        if not positions:
            logging.info("현재 오픈 포지션이 없습니다.")
            return 0.0
        for pos in positions:
            logging.info(f"포지션 데이터: {pos}")
            if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                # 키 목록 출력
                logging.info(f"포지션 데이터 키: {pos.keys()}")
                # 'unrealizedProfit' 키 존재 여부 확인
                if 'unrealizedProfit' in pos:
                    unrealized_pnl = float(pos['unrealizedProfit'])
                elif 'unRealizedProfit' in pos:
                    unrealized_pnl = float(pos['unRealizedProfit'])
                else:
                    logging.error("'unrealizedProfit' 키를 찾을 수 없습니다.")
                    return 0.0
                return unrealized_pnl
        return 0.0
    except BinanceAPIException as e:
        logging.error(f"Binance API 에러: {e}")
        return 0.0
    except Exception as e:
        logging.error(f"포지션 손익 상태 조회 오류: {e}")
        return 0.0


# 포지션 진입 가격 가져오기 함수
def get_position_entry_price(symbol):
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                entry_price = float(pos['entryPrice'])
                return entry_price
        return None
    except Exception as e:
        logging.error(f"진입 가격 조회 오류: {e}")
        return None

# 추가 진입 시 수량 계산 함수
def calculate_additional_quantity(symbol, available_balance, current_price):
    investment_amount = initial_investment * investment_step_percentage / 100
    quantity = investment_amount / current_price
    quantity = format_quantity(symbol, quantity)
    return quantity

# 손절매 도달 시 추가 진입 로직 구현 함수
def monitor_stop_loss_and_add_position(symbol, position_side):
    global current_investment_percentage, entry_count
    # 현재 포지션의 손익 상태 및 가격 정보 가져오기
    unrealized_pnl = get_position_profit_loss(symbol)
    current_price = get_current_price(symbol)
    entry_price = get_position_entry_price(symbol)
    if current_price is None or entry_price is None:
        logging.error("가격 정보를 가져올 수 없어 추가 진입을 할 수 없습니다.")
        return
    # 손절매 가격 계산
    adjusted_stop_loss_percentage = 0.4 + 0.8  # 손절매 비율 + 수수료 비율
    if position_side == SIDE_BUY:
        stop_loss_price = entry_price * (1 - adjusted_stop_loss_percentage / 100)
    else:
        stop_loss_price = entry_price * (1 + adjusted_stop_loss_percentage / 100)
    # 현재 가격이 손절매 가격에 도달했는지 확인
    if (position_side == SIDE_BUY and current_price <= stop_loss_price) or \
       (position_side == SIDE_SELL and current_price >= stop_loss_price):
        logging.info("손절매 가격에 도달하여 추가 진입을 시도합니다.")
        # 최대 진입 횟수 및 자본금 사용률 확인
        if entry_count < max_entry_count and current_investment_percentage + investment_step_percentage <= max_investment_percentage:
            # 추가 진입 수행
            available_balance = get_available_balance()
            additional_quantity = calculate_additional_quantity(symbol, available_balance, current_price)
            place_order(symbol, position_side, additional_quantity, "손절매 도달로 인한 추가 진입")
            set_stop_loss_and_take_profit(symbol, None)
        else:
            # 최대 한도에 도달하여 포지션 청산
            logging.info("최대 진입 횟수 또는 자본금 사용률에 도달하여 포지션을 청산합니다.")
            close_all_positions()

# 자동매매 전략 실행 함수
def run_strategy():
    symbol = 'BTCUSDT'
    desired_leverage = 10

    # 레버리지 설정
    set_leverage(symbol, desired_leverage)

    # 현재 레버리지 확인
    current_leverage = get_current_leverage(symbol)
    if current_leverage != desired_leverage:
        logging.error(f"레버리지가 {desired_leverage}배로 설정되지 않았습니다. 현재 레버리지: {current_leverage}배")
        sys.exit(1)

    # 모든 오픈 포지션 청산
    close_all_positions()

    # 최소 주문 수량 확인
    min_qty, step_size, min_notional = get_min_order_quantity(symbol)
    if min_qty is None or min_notional is None:
        logging.error("최소 주문 수량 또는 최소 명목가를 가져올 수 없어 전략을 중단합니다.")
        sys.exit(1)

    # 총 자본금 로깅
    logging.info(f"전략 시작 - 총 자본금: {initial_investment} USDT")

    while True:
        try:
            # 잔고 확인
            available_balance = get_available_balance()
            if available_balance is None:
                logging.error("잔고 조회 실패")
                time.sleep(300)  # 5분 대기 후 다시 시도
                continue

            # 리스크 관리
            risk_management(available_balance)

            # 현재 가격 확인
            current_price = get_current_price(symbol)
            if current_price is None:
                logging.error("현재 가격 조회 실패")
                time.sleep(300)  # 5분 대기 후 다시 시도
                continue

            # 기술적 지표 확인 (RSI, EMA20, EMA60)
            rsi, ema20, ema60 = get_technical_indicators(symbol)
            if rsi is None or ema20 is None or ema60 is None:
                logging.error("기술적 지표 조회 실패")
                time.sleep(300)  # 5분 대기 후 다시 시도
                continue

            # 매매 조건 설정 (RSI 및 EMA 기준)
            trend = None
            reason = ""
            if rsi < 30 and ema20 > ema60:
                trend = SIDE_BUY
                reason = "RSI가 30 이하로 과매도 상태이고 EMA20이 EMA60 위에 있음 (추세 강세)"
            elif rsi > 70 and ema20 < ema60:
                trend = SIDE_SELL
                reason = "RSI가 70 이상으로 과매수 상태이고 EMA20이 EMA60 아래에 있음 (추세 약세)"
            elif ema20 > ema60:
                trend = SIDE_BUY
                reason = "EMA20이 EMA60을 상향 돌파 (추세 추종)"
            elif ema20 < ema60:
                trend = SIDE_SELL
                reason = "EMA20이 EMA60을 하향 돌파 (추세 추종)"

            if not trend:
                logging.info("매매 신호가 감지되지 않았습니다.")
                time.sleep(300)
                continue

            # 동적 투자 비율 계산
            investment_percentage = calculate_dynamic_investment_percentage(rsi, ema20, ema60)
            # 동적 투자 비율을 기반으로 수량 계산 (최소 명목가를 충족하도록)
            dynamic_quantity = (initial_investment * investment_percentage / 100) / current_price
            dynamic_quantity = max(dynamic_quantity, min_notional / current_price, min_qty)
            dynamic_quantity = format_quantity(symbol, dynamic_quantity)  # 소수점 맞춤

            # 포지션 진입
            order = place_order(symbol, trend, dynamic_quantity, reason)
            set_stop_loss_and_take_profit(symbol, order)

            # 매물대 매매 기법 추가 (매물대에서 지지/저항 여부 판단)
            klines = client.futures_klines(symbol=symbol, interval='5m', limit=200)
            close_prices = [float(kline[4]) for kline in klines]
            volume_profile = pd.Series(close_prices).value_counts().sort_values(ascending=False)
            if not volume_profile.empty:
                major_price = volume_profile.index[0]  # 가장 많이 거래된 가격 (매물대)
                if major_price * 0.99 < current_price < major_price * 1.01:
                    if trend == SIDE_BUY:
                        reason += f" | 매물대 근처에서 반등 매수 시도 (투자 비율: {investment_percentage}%)"
                        order = place_order(symbol, SIDE_BUY, dynamic_quantity, reason)
                        set_stop_loss_and_take_profit(symbol, order)
                    elif trend == SIDE_SELL:
                        reason += f" | 매물대 근처에서 돌파 매도 시도 (투자 비율: {investment_percentage}%)"
                        order = place_order(symbol, SIDE_SELL, dynamic_quantity, reason)
                        set_stop_loss_and_take_profit(symbol, order)

            # 열린 포지션이 있는 경우 추가 진입 로직 실행
            open_positions = get_open_positions()
            if open_positions:
                position_side = SIDE_BUY if float(open_positions[0]['positionAmt']) > 0 else SIDE_SELL
                # 손절매 도달 시 추가 진입 로직 실행
                monitor_stop_loss_and_add_position(symbol, position_side)
                # 최대 진입 횟수 또는 자본금 사용률 도달 시 포지션 청산
                if entry_count >= max_entry_count or current_investment_percentage >= max_investment_percentage:
                    logging.info("최대 진입 횟수 또는 최대 자본금 사용률에 도달하여 포지션을 청산합니다.")
                    close_all_positions()

            time.sleep(300)  # 5분 대기 후 다음 루프

        except KeyboardInterrupt:
            logging.info("스크립트를 수동으로 종료했습니다.")
            break
        except Exception as e:
            logging.error(f"전략 실행 중 오류 발생: {e}")
            logging.error(traceback.format_exc())
            time.sleep(600)  # 오류 발생 시 10분 대기 후 다시 시도

if __name__ == "__main__":
    run_strategy()
