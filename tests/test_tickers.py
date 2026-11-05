import pandas as pd

from classic_patterns.data.tickers import filter_universe, normalize_listings, parse_symdir
from classic_patterns.utils.config import load_config

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZVZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N
ABCDW|Some Corp. - Warrants|S|N|N|100|N|N
ABCDU|Some Acquisition Corp. - Units|S|N|N|100|N|N
SPCA|Some Acquisition Corp. - Class A Ordinary Share|S|N|N|100|N|N
BZ|KANZHUN LIMITED - American Depository Shares|Q|N|N|100|N|N
File Creation Time: 1023202621:31|||||||
"""

OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK.B
BAC$K|Bank of America Corporation Depositary Shares Preferred Stock, Series HH|N|BACpK|N|100|N|BAC-K
SCE$M|SCE Trust VII 7.50% Trust Preference Securities|N|SCEpM|N|100|N|SCE-M
EPD|Enterprise Products Partners L.P. Common Units|N|EPD|N|100|N|EPD
PRT|PermRock Royalty Trust Units of Beneficial Interest|N|PRT|N|100|N|PRT
AMH|American Homes 4 Rent Common Shares of Beneficial Interest|N|AMH|N|100|N|AMH
BME|Blackrock Health Sciences Trust|N|BME|N|100|N|BME
FBRT|Franklin BSP Realty Trust, Inc. Common Stock|N|FBRT|N|100|N|FBRT
KKRT|KKR & Co. Inc. 6.875% Subordinated Notes due 2065|N|KKRT|N|100|N|KKRT
CBOE|Cboe Global Markets, Inc. Common Stock|Z|CBOE|N|100|N|CBOE
VXX|iPath Series B S&P 500 VIX Short-Term Futures ETN|Z|VXX|N|100|N|VXX
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
ASML|ASML Holding N.V. New York Registry Shares|N|ASML|N|100|N|ASML
File Creation Time: 1023202621:31||||||
"""


def run():
    listings = normalize_listings(parse_symdir(NASDAQ), parse_symdir(OTHER))
    kept, dropped = filter_universe(listings, load_config("universe"))
    return kept.set_index("symbol"), dropped.set_index("symbol")["reason"]


def test_kept_and_dropped():
    kept, reasons = run()
    assert set(kept.index) == {
        "AAPL",
        "SPCA",
        "BZ",
        "BRK.B",
        "EPD",
        "PRT",
        "AMH",
        "FBRT",
        "CBOE",
        "ASML",
    }
    assert reasons.to_dict() == {
        "QQQ": "etf",
        "ZVZZT": "test_issue",
        "ABCDW": "warrant",
        "ABCDU": "unit",
        "BACpK": "preferred_symbol",
        "SCEpM": "preferred_symbol",
        "BME": "closed_end_fund",
        "KKRT": "note_debenture",
        "VXX": "exchange_Z",
        "SPY": "etf",
    }


def test_flags_and_yahoo_symbol():
    kept, _ = run()
    assert kept.loc["BRK.B", "yahoo_symbol"] == "BRK-B"
    assert kept.loc["BZ", "is_adr"] and kept.loc["ASML", "is_adr"]
    assert not kept.loc["AAPL", "is_adr"]
    assert kept.loc["SPCA", "is_spac"] and not kept.loc["AAPL", "is_spac"]
    assert isinstance(kept, pd.DataFrame)
