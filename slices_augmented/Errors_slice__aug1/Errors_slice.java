/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        try {
            return null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }

    @Positive
    int[] arr = new int[5];

    // unsafe
    @Positive
    @GTENegativeOne int n1p = -1;
    @Positive
    @LowerBoundUnknown int u = -10;

    // safe
    @Positive
    @NonNegative int nn = 0;
    @Positive
    @Positive int p = 1;

    // :: error: (array.access.unsafe.low)
    @Positive
    int a = arr[n1p];

    // :: error: (array.access.unsafe.low)
    @Positive
    int b = arr[u];

    @Positive
    int c = arr[nn];
    @Positive
    int d = arr[p];
   
        for (int __cfwr_i38 = 0; __cfwr_i38 < 3; __cfwr_i38++) {
            double __cfwr_data67 = 74.97;
        }
 @Positive
  }

    public static Character __cfwr_temp999(Object __cfwr_p0, Float __cfwr_p1, short __cfwr_p2) {
        try {
            try {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 5; __cfwr_i31++) {
            boolean __cfwr_var16 = true;
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        return null;
    }
    protected static Double __cfwr_aux925(byte __cfwr_p0) {
        Boolean __cfwr_node91 = null;
        return null;
    }
}