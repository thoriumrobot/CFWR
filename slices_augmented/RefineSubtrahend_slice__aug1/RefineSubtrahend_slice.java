/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineSubtrahend_slice {
    @Positive
  void withConstant(int[] a, @NonNegative int l) {
        try {
            return -23.86;
        } catch (Exception __cfwr_e11) {
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

    public Float __cfwr_helper133() {
        Boolean __cfwr_obj61 = null;
        while (true) {
            if (((318 >> true) >> null) && false) {
            try {
            if (false && false) {
            Long __cfwr_entry6 = null;
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            while (true) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 6; __cfwr_i44++) {
            while (true) {
            try {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 6; __cfwr_i93++) {
            return 951L;
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        Float __cfwr_var4 = null;
        return null;
    }
}