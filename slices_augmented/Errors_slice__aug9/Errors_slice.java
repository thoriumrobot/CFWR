/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        String __cfwr_var94 = "data44";

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
    @Positiv
        try {
            if (true || false) {
            while (false) {
            return -75.49f;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
e
  }

    private byte __cfwr_aux328() {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 5; __cfwr_i51++) {
            Boolean __cfwr_temp97 = null;
        }
        return null;
    }
}