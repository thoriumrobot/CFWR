/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
  void test() {
        try {
            if (false && true) {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 8; __cfwr_i13++) {
            try {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 9; __cfwr_i7++) {
            try {
            while (false) {
            while (true) {
            try {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 2; __cfwr_i9++) {
            short __cfwr_obj51 = null;
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }

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

    short __cfwr_calc402() {
        if (((-328L ^ null) % ('U' >> false)) && (91.68f % null)) {
            try {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 1; __cfwr_i84++) {
            return null;
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
        return (-65.55 | (null | -2.23f));
    }
}