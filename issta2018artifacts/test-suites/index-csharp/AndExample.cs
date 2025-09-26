using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class AndExample {

    /*@SuppressWarnings("index")*/ // forward reference to field iYearInfoCache
    private static readonly int CACHE_SIZE = 1 << 10;

    private static readonly int CACHE_MASK = CACHE_SIZE - 1;

    private static readonly String[] iYearInfoCache = new String[CACHE_SIZE];

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(CACHE_SIZE >= 0 && CACHE_SIZE <= iYearInfoCache.Length);
        Contract.Invariant(CACHE_MASK >= 0 && CACHE_MASK < iYearInfoCache.Length);
    }

    private String getYearInfo(int year)
    {
        Contract.Requires(CACHE_SIZE >= 0 && CACHE_SIZE <= iYearInfoCache.Length);
        Contract.Requires(CACHE_MASK >= 0 && CACHE_MASK < iYearInfoCache.Length);
        return iYearInfoCache[year & CACHE_MASK];
    }
}
