/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineSubtrahend_slice {
    @Positive
  void withConstant(int[] a, @NonNegative int l) {
        Object __cfwr_node53 = null;

    
        Double __cfwr_temp51 = null;
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

    public static short __cfwr_aux255(byte __cfwr_p0, Object __cfwr_p1, Integer __cfwr_p2) {
        try {
            String __cfwr_obj25 = "result33";
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        for (int __cfwr_i48 = 0; __cfwr_i48 < 9; __cfwr_i48++) {
            Double __cfwr_result75 = null;
        }
        return null;
    }
    private char __cfwr_process420() {
        try {
            Long __cfwr_var60 = null;
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        Character __cfwr_entry33 = null;
        while (true) {
            while ((752L << (-46.89 + 78.44))) {
            while ((246 ^ null)) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 1; __cfwr_i5++) {
            return (null - null);
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 'Y';
    }
}