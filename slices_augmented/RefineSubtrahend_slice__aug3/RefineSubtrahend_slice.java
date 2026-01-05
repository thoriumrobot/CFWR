/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineSubtrahend_slice {
    @Positive
  void withConstant(int[] a, @NonNegative int l) {
        try {
            Long __cfwr_node65 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }

    @Positive
    if (a.length - l > 10) {
    @Positive
      int x = a[l + 10];
    @Positive
    }
    @Positive
    if (a.length - 10 > l) {
    @Positive
      int x = a[l + 10];
    @Positive
    }
    @Positive
    if (a.length - l >= 10) {
      // :: error: (array.access.unsafe.high)
    @Positive
      int x = a[l + 10];
    @Positive
      int x1 = a[l + 9];
    @Positive
    }
    @Positive
  }

    @Positive
  void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
    @Positive
    if (a.length - l > j) {
    @Positive
      if (k <= j) {
    @Positive
        int x = a[l + k];
    @Positive
      }
    @Positive
    }
    @Positive
    if (a.length - j > l) {
    @Positive
      if (k <= j) {
    @Positive
        int x = a[l + k];
    @Positive
      }
    @Positive
    }
    @Positive
    if (a.length - j >= l) {
    @Positive
      if (k <= j) {
        // :: error: (array.access.unsafe.high)
    @Positive
        int x = a[l + k];
        // :: error: (array.access.unsafe.low)
    @Positive
        int x1 = a[l + k - 1];
    @Positive
      }
    @Positive
    }
    @Positive
  }

    boolean __cfwr_func423(double __cfwr_p0, boolean __cfwr_p1) {
        Float __cfwr_item65 = null;
        char __cfwr_var14 = '0';
        for (int __cfwr_i23 = 0; __cfwr_i23 < 3; __cfwr_i23++) {
            try {
            return null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        return false;
    }
    private static Float __cfwr_process76(int __cfwr_p0, String __cfwr_p1) {
        return null;
        for (int __cfwr_i88 = 0; __cfwr_i88 < 9; __cfwr_i88++) {
            long __cfwr_item58 = -720L;
        }
        try {
            Boolean __cfwr_data35 = null;
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        return null;
    }
}