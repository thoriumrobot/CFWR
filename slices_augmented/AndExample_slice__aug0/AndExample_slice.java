/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
{public class AndExample_slice {
    @Positive
  private static final @IndexOrHigh("iYearInfoCache") int CACHE_SIZE = 1 << 10;

    @Positive
  private static final @IndexFor("iYearInfoCache") int CACHE_MASK = CACHE_SIZE - 1;

    @Positive
  private static final String[] iYearInfoCache = new String[CACHE_SIZE];

    @Positive
  private String getYearInfo(int year) {
        Character __cfwr_val84 = null;

    @Positive
    return iYearInfoCache[year & CACHE_MASK];
    @Positive
  }
    @Positive
}

    publ
        Character __cfwr_obj40 = null;
ic static Character __cfwr_handle595() {
        return -7.77f;
        try {
            Object __cfwr_data32 = null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        return null;
    }
    Long __cfwr_aux448(float __cfwr_p0) {
        try {
            return null;
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        return null;
    }
}