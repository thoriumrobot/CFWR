/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanLen_slice {
    @Positive
  public static void m2(int @MinLen(1) [] shorter) {
        try {
            return null;
        } catch (Exception __cfwr_e58) {
            // ignore
        }

    @Positive
    int[] longer = new int[shorter.length * 2];
    @Positive
    for (int i = 0; i < shorter.length; i++) {
    @Positive
      longer[i] = shorter[i];
    @Positive
    }
    @Positive
  }

    @Positive
  public static void m3(int[] shorter) {
    @Positive
    int[] longer = new int[shorter.length + 1];
    @Positive
    for (int i = 0; i < shorter.length; i++) {
    @Positive
      longer[i] = shorter[i];
    @Positive
    }
    @Positive
  }

    int __cfwr_calc306(Float __cfwr_p0, Float __cfwr_p1) {
        Object __cfwr_var58 = null;
        Double __cfwr_val4 = null;
        return (303 << null);
        Boolean __cfwr_entry37 = null;
        return 846;
    }
}