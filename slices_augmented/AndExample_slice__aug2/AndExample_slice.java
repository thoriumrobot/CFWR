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
        Double __cfwr_entry76 = null;

    @Positive
    return iYearInfoCache[year & CACHE_MASK];
    @Positive
  }
    @Positive
}

    static long __cfwr_proc829(char __cfwr_p0, String __cfwr_p1, short __cfwr_p2) {
        return null;
        return null;
        return 198L;
    }
    private static Boolean __cfwr_temp672(short __cfwr_p0) {
        Integer __cfwr_var59 = null;
        return null;
        try {
            while (false) {
            int __cfwr_entry14 = -357;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        char __cfwr_node17 = 'u';
        return null;
    }
}