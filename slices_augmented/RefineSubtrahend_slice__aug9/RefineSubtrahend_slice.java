/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineSubtrahend_slice {
    @Positive
  void withConstant(int[] a, @NonNegative int l) {
        try {
            return 5.49;
        } catch (Exception __cfwr_e20) {
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

    public static Character __cfwr_handle220(Float __cfwr_p0, String __cfwr_p1, char __cfwr_p2) {
        for (int __cfwr_i9 = 0; __cfwr_i9 < 2; __cfwr_i9++) {
            return false;
        }
        return null;
    }
}