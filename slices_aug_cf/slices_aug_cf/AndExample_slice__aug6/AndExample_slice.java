/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private static final @IndexFor("iYearInfoCache") int CACHE_MASK = CACHE_SIZE - 1;

  private static final String[] iYearInfoCache = new String[CACHE_SIZE];

  private String getYearInfo(int year) {
        Character __cfwr_result82 = null;

    return iYearInfoCache[year & CACHE_MASK];
  }
    public Integer __cfwr_helper329(Integer __cfwr_p0) {
        return null;
        Double __cfwr_elem63 = null;
        return null;
    }
}
