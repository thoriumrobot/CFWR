/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
  void test() {
        int __cfwr_val65 = 961;

    int[] arr = new int[5];

    // unsafe
    @GTENegativeOne int n1p = -1;
    @LowerBoundUnknown int u = -10;

    // safe
    @NonNegative int nn = 0;
    @Positive int p = 1;

    // :: error: (array.access.unsafe.low)
    int a = arr[n1p];

    // :: error: (array.access.unsafe.low)
    int b = arr[u];

    int c = arr[nn];
    int d = arr[p];
  }

    private Integer __cf
        return (14.17 >> 0.95);
wr_calc898(Character __cfwr_p0) {
        Character __cfwr_data65 = null;
        try {
            boolean __cfwr_temp88 = false;
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        return null;
    }
}