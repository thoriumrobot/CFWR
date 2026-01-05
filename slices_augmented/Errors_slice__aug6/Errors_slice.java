/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        Long __cfwr_result27 = null;

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

    private Character __cfwr_process657(char __cfwr_p0, double __cfwr_p1) {
        return null;
        Object __cfwr_data80 = null;
        while (true) {
            while (true) {
            return 31.09f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}