/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 5; __cfwr_i70++) {
            try {
            int __cfwr_var2 = (-43.36 >> null);
        } catch (Exception __cfwr_e5) {
            // ignore
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

    static double __cfwr_proc113() {
        return false;
        for (int __cfwr_i86 = 0; __cfwr_i86 < 6; __cfwr_i86++) {
            while (true) {
            return -615;
            break; // Prevent infinite loops
        }
        }
        return 'm';
        return -26.38;
    }
    public boolean __cfwr_func654(long __cfwr_p0) {
        while (false) {
            try {
            return -14.87f;
        } catch (Exception __cfwr_e10) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false || true) {
            if ((null >> (false * null)) && false) {
            try {
            try {
            while (false) {
            char __cfwr_val92 = '4';
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        }
        }
        if (false || (-14.29 + null)) {
            try {
            return "hello13";
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        return true;
    }
}