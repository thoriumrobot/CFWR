/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        for (int __cfwr_i24 = 0; __cfwr_i24 < 8; __cfwr_i24++) {
            if (true && (-893L % null)) {
            Integer __cfwr_obj88 = null;
        }
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
    @Positive
  }

    private static long __cfwr_calc864() {
        try {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 8; __cfwr_i41++) {
            String __cfwr_data75 = "value1";
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        return -439L;
    }
    private float __cfwr_util522(char __cfwr_p0) {
        for (int __cfwr_i15 = 0; __cfwr_i15 < 4; __cfwr_i15++) {
            return 'K';
        }
        for (int __cfwr_i8 = 0; __cfwr_i8 < 2; __cfwr_i8++) {
            boolean __cfwr_obj23 = (305L * 85.27f);
        }
        return 85.29f;
    }
}