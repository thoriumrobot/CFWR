/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Errors_slice {
    @Positive
  void test() {
        if (true || (-637 & 'j')) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 1; __cfwr_i64++) {
            if ((75.25f - (false | -697)) && false) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 1; __cfwr_i47++) {
            while (true) {
            if (true && true) {
            return false;
        }
            break; // Prevent infinite loops
        }
        }
        }
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

    private long __cfwr_handle499(int __cfwr_p0) {
        return null;
        return null;
        float __cfwr_obj52 = -43.90f;
        try {
            return ((null | false) & null);
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        return 719L;
    }
    public static Integer __cfwr_helper375(Double __cfwr_p0, Boolean __cfwr_p1) {
        try {
            try {
            Long __cfwr_entry43 = null;
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        int __cfwr_item69 = 506;
        if ((null / 742L) || ((7.09 ^ true) << 626L)) {
            try {
            while (false) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 4; __cfwr_i63++) {
            try {
            return null;
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        return null;
    }
}