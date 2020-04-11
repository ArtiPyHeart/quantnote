import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 2.1 PCA weights from a risk distribution r
def pcaWeights(cov, riskDist=None, riskTarget=1):
    # Following the riskAlloc distribution, match riskTarget
    eVal, eVec = np.linalg.eigh(cov)  # must be Hermitian
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1.0
    loads = riskTarget * (riskDist / eVal) ** 0.5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # ctr=(loads/riskTarget)**2*eVal # verify riskDist
    return wghts


# -----------------------------------------------------------------------------
# 2.2 form a gaps series, detract it from prices
def getRolledSeries(pathIn, key):
    series = pd.read_hdf(pathIn, key="bars/ES_10k")
    series["Time"] = pd.to_datetime(series["Time"], format="%Y%m%d%H%M%S%f")
    series = series.set_index("Time")
    gaps = rollGaps(series)
    for fld in ["Close", "VWAP"]:
        series[fld] -= gaps
    return series


def rollGaps(
    series,
    dictio={"Instrument": "FUT_CUR_GEN_TICKER", "Open": "PX_OPEN", "Close": "PX_LAST"},
    matchEnd=True,
):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio["Instrument"]].drop_duplicates(keep="first").index
    gaps = series[dictio["Close"]] * 0
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in rollDates]  # index of days prior to roll
    gaps.loc[rollDates[1:]] = (
        series[dictio["Open"]].loc[rollDates[1:]]
        - series[dictio["Close"]].iloc[iloc[1:]].values
    )
    gaps = gaps.cumsum()
    if matchEnd:
        gaps -= gaps.iloc[-1]  # roll backward
    return gaps


# -----------------------------------------------------------------------------
# 2.3 non-negative rolled price series
raw = pd.read_csv(filePath, index_col=0, parse_dates=True)
gaps = rollGaps(raw, dictio={"Instrument": "Symbol", "Open": "Open", "Close": "Close"})
rolled = raw.copy(deep=True)
for fld in ["Open", "Close"]:
    rolled[fld] -= gaps
rolled["Returns"] = rolled["Close"].diff() / raw["Close"].shift(1)
rolled["rPrices"] = (1 + rolled["Returns"]).cumprod()

# -----------------------------------------------------------------------------
# 2.4 the symmetric cumsum filter
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# -----------------------------------------------------------------------------
# 3.1 daily volatility estimates
def getDailyVol(close, span0=100):
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    except Exception as e:
        print(f"error: {e}\nplease confirm no duplicate indices")
    df0 = df0.ewm(span=span0).std().rename("dailyVol")
    return df0


# -----------------------------------------------------------------------------
# 3.2 triple-barrier labeling method
def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_["t1"].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


# -----------------------------------------------------------------------------
# 3.3 getting the time of first touch - computationally expensive
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    side_ = pd.Series(1.0, index=trgt.index)
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )
    df0 = mpPandasObj(
        func=applyPtSlOnT1,
        pdObj=("molecule", events.index),
        numThreads=numThreads,
        close=close,
        events=events,
        ptSl=[ptSl, ptSl],
    )
    events["t1"] = df0.dropna(how="all").min(axis=1)  # pd.min ignores nan
    events = events.drop("side", axis=1)
    return events


# -----------------------------------------------------------------------------
# 3.4 adding a vertical barrier
t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
t1 = t1[t1 < close.shape[0]]
t1 = pd.Series(close.index[t1], index=tEvents[: t1.shape[0]])  # NaNs at end

# -----------------------------------------------------------------------------
# 3.5 labeling for side and size
def getBins(events, close):
    # 1) prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    out["bin"] = np.sign(out["ret"])
    return out


# -----------------------------------------------------------------------------
# 3.6 expanding getEventsto incorporate meta-labeling
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1.0, index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )
    df0 = mpPandasObj(
        func=applyPtSlOnT1,
        pdObj=("molecule", events.index),
        numThreads=numThreads,
        close=inst["Close"],
        events=events,
        ptSl=ptSl_,
    )
    events["t1"] = df0.dropna(how="all").min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop("side", axis=1)
    return events


# -----------------------------------------------------------------------------
# 3.7 expanding getBins to incorporate meta-labeling
def getBins(events, close):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    —events.index is event's starttime
    —events[’t1’] is event's endtime
    —events[’trgt’] is event's target
    —events[’side’] (optional) implies the algo's position side
    Case 1: (’side’ not in events): bin in (-1,1) <—label by price action
    Case 2: (’side’ in events): bin in (0,1) <—label by pnl (meta-labeling)
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"] *= events_["side"]  # meta-labeling
    out["bin"] = np.sign(out["ret"])
    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0  # meta-labeling
    return out


# -----------------------------------------------------------------------------
# 3.8 dropping under-populated labels
def dropLabels(events, minPct=0.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print("dropped label", df0.argmin(), df0.min())
        events = events[events["bin"] != df0.argmin()]
    return events


# -----------------------------------------------------------------------------
# 4.1 estimating the uniqueness of a label
def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    """
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[
        : t1[molecule].max()
    ]  # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0] : iloc[1] + 1])
    for tIn, tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1.0
    return count.loc[molecule[0] : t1[molecule].max()]


# -----------------------------------------------------------------------------
# 4.2 estimating the average uniqueness of a label
def mpSampleTW(t1, numCoEvents, molecule):
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1.0 / numCoEvents.loc[tIn:tOut]).mean()
    return wght


numCoEvents = mpPandasObj(
    mpNumCoEvents,
    ("molecule", events.index),
    numThreads,
    closeIdx=close.index,
    t1=events["t1"],
)
numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep="last")]
numCoEvents = numCoEvents.reindex(close.index).fillna(0)
out["tW"] = mpPandasObj(
    mpSampleTW,
    ("molecule", events.index),
    numThreads,
    t1=events["t1"],
    numCoEvents=numCoEvents,
)

# -----------------------------------------------------------------------------
# 4.3 buiding an indicator matrix
def getIndMatrix(barIx, t1):
    # Get indicator matrix
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1.0
    return indM


# -----------------------------------------------------------------------------
# 4.4 compute average uniqueness
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness
    return avgU


# ------------------------------------------------------------------------------
# 4.5 return sample from sequential bootstrap
def seqBootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


# ------------------------------------------------------------------------------
# 4.6 example of sequential bootstrap
def main():
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])  # t0,t1 for each feature obs
    barIx = range(t1.max() + 1)  # index of bars
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print(phi)
    print("Standard uniqueness:", getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print(phi)
    print("Sequential uniqueness:", getAvgUniqueness(indM[phi]).mean())


# ------------------------------------------------------------------------------
# 4.7 generating a random t1 series
def getRndT1(numObs, numBars, maxH):
    # random t1 Series
    t1 = pd.Series()
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


# -----------------------------------------------------------------------------
# 4.8 uniqueness from standard and sequential bootstraps
def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {"stdU": stdU, "seqU": seqU}


# -----------------------------------------------------------------------------
# 4.9 muiti-threaded monte carlo
from mpEngine import processJobs, processJobs_


def mainMC(numObs=10, numBars=100, maxH=5, numIters=1e6, numThreads=24):
    # Monte Carlo experiments
    jobs = []
    for i in range(int(numIters)):
        job = {"func": auxMC, "numObs": numObs, "numBars": numBars, "maxH": maxH}
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    print(pd.DataFrame(out).describe())


# -----------------------------------------------------------------------------
# 4.10 determination of sample weight by absolute return attribution
def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


out["w"] = mpPandasObj(
    mpSampleW,
    ("molecule", events.index),
    numThreads,
    t1=events["t1"],
    numCoEvents=numCoEvents,
    close=close,
)
out["w"] *= out.shape[0] / out["w"].sum()
# -----------------------------------------------------------------------------
# 4.11 implementation of time-decay factors
def getTimeDecay(tW, clfLastW=1.0):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print(const, slope)
    return clfW


# -----------------------------------------------------------------------------
# 5.1 weighting function
def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how="outer")
    ax = w.plot()
    ax.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    plotWeights(dRange=[0, 1], nPlots=11, size=6)
    plotWeights(dRange=[1, 2], nPlots=11, size=6)

# -----------------------------------------------------------------------------
# 5.2 standard fracdiff (expanding window)
def fracDiff(series, d, thres=0.01):
    """
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1) :, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# -----------------------------------------------------------------------------
# 5.3 the new fixed-width window fracdiff method
def getWeights_FFD(d, thres):
    w, k = [1.0], 1
    while True:
        w_ = w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(series, d, thres=1e-5):
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# -----------------------------------------------------------------------------
# 5.4 finding the minimum D value that passes the ADF test
def plotMinFFD():
    from statsmodels.tsa.stattools import adfuller

    path, instName = "./", "ES1_Index_Method12"
    out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "corr"])
    df0 = pd.read_csv(path + instName + ".csv", index_col=0, parse_dates=True)
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[["Close"]]).resample("1D").last()  # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres=0.01)
        corr = np.corrcoef(df1.loc[df2.index, "Close"], df2["Close"])[0, 1]
        df2 = adfuller(df2["Close"], maxlag=1, regression="c", autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]["5%"]] + [corr]  # with critical value
    out.to_csv(path + instName + "_testMinFFD.csv")
    out[["adfStat", "corr"]].plot(secondary_y="adfStat")
    plt.axhline(out["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    plt.savefig(path + instName + "_testMinFFD.png")


# -----------------------------------------------------------------------------
# 6.2 three ways of setting up an RandomForest
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

clf0 = RandomForestClassifier(
    n_estimators=1000, class_weight="balanced_subsample", criterion="entropy"
)

clf1 = DecisionTreeClassifier(
    criterion="entropy", max_features="auto", class_weight="balanced"
)
clf1 = BaggingClassifier(
    base_estimator=clf1, n_estimators=1000, max_samples=avgU
)  # average uniqueness between samples

clf2 = RandomForestClassifier(
    n_estimators=1,
    criterion="entropy",
    bootstrap=False,
    class_weight="balanced_subsample",
)
clf2 = BaggingClassifier(
    base_estimator=clf2, n_estimators=1000, max_samples=avgU, max_features=1.0
)

# -----------------------------------------------------------------------------
# 7.1 purging observation in the training set
def getTrainTimes(t1, testTimes):
    """
    Given testTimes, find the times of the training observations.
    —t1.index: Time when the observation started.
    —t1.value: Time when the observation ended.
    —testTimes: Times of testing observations.
    """
    trn = t1.copy(deep=True)
    for i, j in testTimes.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index  # train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index  # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


# -----------------------------------------------------------------------------
# 7.2 embargo on training observations
def getEmbargoTimes(times, pctEmbargo):
    # Get embargo time for each bar
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))
    return mbrg


testTimes = pd.Series(mbrg[dt1], index=[dt0])  # include embargo before purge
trainTimes = getTrainTimes(t1, testTimes)
testTimes = t1.loc[dt0:dt1].index

# -----------------------------------------------------------------------------
# 7.3 cross-validation class when observations overlap
class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx + mbrg :])
                )
            yield train_indices, test_indices


# -----------------------------------------------------------------------------
# 7.4 using the PurgedKFold class
def cvScore(
    clf,
    X,
    y,
    sample_weight,
    scoring="neg_log_loss",
    t1=None,
    cv=None,
    cvGen=None,
    pctEmbargo=None,
):
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")
    from sklearn.metrics import log_loss, accuracy_score
    from clfSequential import PurgedKFold

    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(
                y.iloc[test],
                prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values
            )
        score.append(score_)
    return np.array(score)


# -----------------------------------------------------------------------------
# 8.2 MDI feature importance
def featImpMDI(fit, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat(
        {"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1
    )
    imp /= imp["mean"].sum()
    return imp


# -----------------------------------------------------------------------------
# 8.3 MDA feature importance
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring="neg_log_loss"):
    # feat importance based on OOS score reduction
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")
    from sklearn.metrics import log_loss, accuracy_score

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(
                y1, prob, sample_weight=w1.values, labels=clf.classes_
            )
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(
                    y1, prob, sample_weight=w1.values, labels=clf.classes_
                )
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)
    imp = pd.concat(
        {"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1
    )
    return imp, scr0.mean()


# -----------------------------------------------------------------------------
# 8.4 implementation of SFI
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=["mean", "std"])
    for featName in featNames:
        df0 = cvScore(
            clf,
            X=trnsX[[featName]],
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen,
        )
        imp.loc[featName, "mean"] = df0.mean()
        imp.loc[featName, "std"] = df0.std() * df0.shape[0] ** -0.5
    return imp


# -----------------------------------------------------------------------------
# 8.5 computation of orthogonal features
def get_eVec(dot, varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    # 2) only positive eVals
    eVal = pd.Series(eVal, index=["PC_" + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    # 3) reduce dimension, form PCs
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[: dim + 1], eVec.iloc[:, : dim + 1]
    return eVal, eVec


def orthoFeats(dfX, varThres=0.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


# -----------------------------------------------------------------------------
# 8.6 computation of weighted Kendall's tau between feature importance and inverse PCA ranking
import numpy as np
from scipy.stats import weightedtau

featImp = np.array([0.55, 0.33, 0.07, 0.05])  # feature importance
pcRank = np.array([1, 2, 4, 3])  # PCA rank
print(weightedtau(featImp, pcRank ** -1.0)[0])

# -----------------------------------------------------------------------------
# 8.7 creating a synthetic dataset
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification

    trnsX, cont = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=0,
        shuffle=False,
    )
    df0 = pd.DatetimeIndex(
        periods=n_samples, freq=pd.tseries.offsets.BDay(), end=pd.datetime.today()
    )
    trnsX, cont = (pd.DataFrame(trnsX, index=df0),)
    pd.Series(cont, index=df0).to_frame("bin")
    df0 = ["I_" + str(i) for i in range(n_informative)] + [
        "R_" + str(i) for i in range(n_redundant)
    ]
    df0 += ["N_" + str(i) for i in range(n_features - len(df0))]
    trnsX.columns = df0
    cont["w"] = 1.0 / cont.shape[0]
    cont["t1"] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


# -----------------------------------------------------------------------------
# 8.8 calling feature importance for any method
def featImportance(
    trnsX,
    cont,
    n_estimators=1000,
    cv=10,
    max_samples=1.0,
    numThreads=24,
    pctEmbargo=0,
    scoring="accuracy",
    method="SFI",
    minWLeaf=0.0,
    **kargs,
):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from mpEngine import mpPandasObj

    n_jobs = -1 if numThreads > 1 else 1  # run 1 thread with ht_helper in dirac1
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=minWLeaf,
    )
    clf = BaggingClassifier(
        base_estimator=clf,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs,
    )
    fit = clf.fit(X=trnsX, y=cont["bin"], sample_weight=cont["w"].values)
    oob = fit.oob_score_
    if method == "MDI":
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(
            clf,
            X=trnsX,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pctEmbargo=pctEmbargo,
            scoring=scoring,
        ).mean()
    elif method == "MDA":
        imp, oos = featImpMDA(
            clf,
            X=trnsX,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pctEmbargo=pctEmbargo,
            scoring=scoring,
        )
    elif method == "SFI":
        cvGen = PurgedKFold(n_splits=cv, t1=cont["t1"], pctEmbargo=pctEmbargo)
        oos = cvScore(
            clf,
            X=trnsX,
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen,
        ).mean()
        clf.n_jobs = 1  # paralellize auxFeatImpSFI rather than clf
        imp = mpPandasObj(
            auxFeatImpSFI,
            ("featNames", trnsX.columns),
            numThreads,
            clf=clf,
            trnsX=trnsX,
            cont=cont,
            scoring=scoring,
            cvGen=cvGen,
        )
    return imp, oob, oos


# -----------------------------------------------------------------------------
# 8.9 calling all components
def testFunc(
    n_features=40,
    n_informative=10,
    n_redundant=10,
    n_estimators=1000,
    n_samples=10000,
    cv=10,
):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features—n_informative—n_redundant
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {
        "minWLeaf": [0.0],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.0],
    }
    jobs, out = (dict(izip(dict0, i)) for i in product(*dict0.values())), []
    kargs = {
        "pathOut": "./testFunc/",
        "n_estimators": n_estimators,
        "tag": "testFunc",
        "cv": cv,
    }
    for job in jobs:
        job["simNum"] = (
            job["method"]
            + "_"
            + job["scoring"]
            + "_"
            + "%.2f" % job["minWLeaf"]
            + "_"
            + str(job["max_samples"])
        )
    print
    job["simNum"]
    kargs.update(job)
    imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)
    plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)
    df0 = imp[["mean"]] / imp["mean"].abs().sum()
    df0["type"] = [i[0] for i in df0.index]
    df0 = df0.groupby("type")["mean"].sum().to_dict()
    df0.update({"oob": oob, "oos": oos})
    df0.update(job)
    out.append(df0)
    out = pd.DataFrame(out).sort_values(
        ["method", "scoring", "minWLeaf", "max_samples"]
    )
    out = out[
        "method", "scoring", "minWLeaf", "max_samples", "I", "R", "N", "oob", "oos"
    ]
    out.to_csv(kargs["pathOut"] + "stats.csv")


# -----------------------------------------------------------------------------
# 8.10 feature importance plotting function
def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0] / 5.0))
    imp = imp.sort_values("mean", ascending=True)
    ax = imp["mean"].plot(
        kind="barh", color="b", alpha=0.25, xerr=imp["std"], error_kw={"ecolor": "r"}
    )
    if method == "MDI":
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1.0 / imp.shape[0], linewidth=1, color="r", linestyle="dotted")
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(
            i.get_width() / 2,
            i.get_y() + i.get_height() / 2,
            j,
            ha="center",
            va="center",
            color="black",
        )
    plt.title(
        "tag="
        + tag
        + " | simNum="
        + str(simNum)
        + " | oob="
        + str(round(oob, 4))
        + " | oos="
        + str(round(oos, 4))
    )
    plt.savefig(pathOut + "featImportance_" + str(simNum) + ".png", dpi=100)
    plt.clf()
    plt.close()


# -----------------------------------------------------------------------------
# 9.1 grid search with purged k-fold cross validation
from sklearn.pipeline import Pipeline


def clfHyperFit(
    feat,
    lbl,
    t1,
    pipe_clf,
    param_grid,
    cv=3,
    bagging=[0, None, 1.0],
    n_jobs=-1,
    pctEmbargo=0,
    **fit_params,
):
    if set(lbl.values) == {0, 1}:
        scoring = "f1"  # f1 for meta-labeling
    else:
        scoring = "neg_log_loss"  # symmetric towards all cases
    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    gs = GridSearchCV(
        estimator=pipe_clf,
        param_grid=param_grid,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=n_jobs,
        iid=False,
    )
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # pipeline
    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(
            base_estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        gs = gs.fit(
            feat,
            lbl,
            sample_weight=fit_params[
                gs.base_estimator.steps[-1][0] + "__sample_weight"
            ],
        )
        gs = Pipeline([("bag", gs)])
    return gs


# -----------------------------------------------------------------------------
# 9.2 an enhanced pipeline class
class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


# -----------------------------------------------------------------------------
# 9.3 randomized search with purged k-fold cv
from sklearn.model_selection import RandomizedSearchCV


def clfHyperFit(
    feat,
    lbl,
    t1,
    pipe_clf,
    param_grid,
    cv=3,
    bagging=[0, None, 1.0],
    rndSearchIter=0,
    n_jobs=-1,
    pctEmbargo=0,
    **fit_params,
):
    if set(lbl.values) == {0, 1}:
        scoring = "f1"  # f1 for meta-labeling
    else:
        scoring = "neg_log_loss"  # symmetric towards all cases
    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    if rndSearchIter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            iid=False,
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            iid=False,
            n_iter=rndSearchIter,
        )
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # pipeline
    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(
            base_estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        gs = gs.fit(
            feat,
            lbl,
            sample_weight=fit_params[
                gs.base_estimator.steps[-1][0] + "__sample_weight"
            ],
        )
        gs = Pipeline([("bag", gs)])
    return gs


# -----------------------------------------------------------------------------
# 9.4 the logUniform_gen class
from scipy.stats import rv_continuous, kstest


class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1, b=np.exp(1)):
    return logUniform_gen(a=a, b=b, name="logUniform")


a, b, size = 1e-3, 1e3, 10000
vals = logUniform(a=a, b=b).rvs(size=size)
print(kstest(rvs=np.log(vals), cdf="uniform", args=(np.log(a), np.log(b / a)), N=size))
print(pd.Series(vals).describe())
plt.subplot(121)
pd.Series(np.log(vals)).hist()
plt.subplot(122)
pd.Series(vals).hist()
plt.show()

# -----------------------------------------------------------------------------
# 10.1 from probabilities to bet size
def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    # get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob - 1.0 / numClasses) / (prob * (1.0 - prob)) ** 0.5  # t-value of OvR
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal=side*size
    if "side" in events:
        signal0 *= events.loc[signal0.index, "side"]  # meta-labeling
    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame("signal").join(events[["t1"]], how="left")
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


# -----------------------------------------------------------------------------
# 10.2 bets are averaged as long as they are still active
def avgActiveSignals(signals, numThreads):
    # compute the average signal among those active
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals["t1"].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(
        mpAvgActiveSignals, ("molecule", tPnts), numThreads, signals=signals
    )
    return out


def mpAvgActiveSignals(signals, molecule):
    """
    At time loc, average signal among those still active.
    Signal is active if:
    a) issued before or at loc AND
    b) loc before signal's endtime, or endtime is still unknown (NaT).
    """
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & (
            (loc < signals["t1"]) | pd.isnull(signals["t1"])
        )
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, "signal"].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


# -----------------------------------------------------------------------------
# 10.3 size discretization to prevent overtrading
def discreteSignal(signal0, stepSize):
    # discretize signal
    signal1 = (signal0 / stepSize).round() * stepSize  # discretize
    signal1[signal1 > 1] = 1  # cap
    signal1[signal1 < -1] = -1  # floor
    return signal1


# -----------------------------------------------------------------------------
# 10.4 dynamic position size and limit price
def betSize(w, x):
    return x * (w + x ** 2) ** -0.5


def getTPos(w, f, mP, maxPos):
    return int(betSize(w, f - mP) * maxPos)


def invPrice(f, w, m):
    return f - m * (w / (1 - m ** 2)) ** 0.5


def limitPrice(tPos, pos, f, w, maxPos):
    sgn = 1 if tPos >= pos else -1
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += invPrice(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    # 0<alpha<1
    return x ** 2 * (m ** -2 - 1)


def main():
    pos, maxPos, mP, f, wParams = 0, 100, 100, 115, {"divergence": 10, "m": 0.95}
    w = getW(wParams["divergence"], wParams["m"])  # calibrate w
    tPos = getTPos(w, f, mP, maxPos)  # get tPos
    lP = limitPrice(tPos, pos, f, w, maxPos)  # limit price for order
