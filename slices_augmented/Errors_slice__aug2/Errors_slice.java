/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        try {
            return false;
        } catch (Exception __cfwr_e96) {
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
    @Positive
  }

    static Double __cfwr_aux761(long __cfwr_p0, double __cfwr_p1) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 8; __cfwr_i73++) {
            return null;
        }
        return null;
    }
}