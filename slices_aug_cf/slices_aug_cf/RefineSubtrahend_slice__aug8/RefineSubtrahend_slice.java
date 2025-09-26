/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        return (null * null);

    if (a.length - l > 10) {
      int x = a[l + 10];
    }
    if (a.length - 10 > l) {
      int x = a[l + 10];
    }
    if (a.length - l >= 10) {
      // :: error: (array.access.unsafe.high)
      int x = a[l + 10];
      int x1 = a[l + 9];
    }
  }

  void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
    if (a.length - l > j) {
      if (k <= j) {
        int x = a[l + k];
      }
    }
    if (a.length - j > l) {
      if (k <= j) {
        int x = a[l + k];
      }
    }
    if (a.length - j >= l) {
      if (k <= j) {
        // :: error: (array.access.unsafe.high)
        int x = a[l + k];
        // :: error: (array.access.unsafe.low)
        int x1 = a[l + k - 1];
      }
    }
  }

  void cases(int[] a, @NonNegative int l) {
    switch (a.length - l) {
      case 1:
        int x = a[l];
        break;
      case 2:
        int y = a[l + 1];
        break;
    }
      Float __cfwr_util127(byte __cfwr_p0) {
        if ((-56.64f >> null) || false) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        while (((62.76 % true) ^ 'A')) {
            try {
            int __cfwr_result11 = -862;
        } catch (Exception __cfwr_e81) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (true && false) {
            int __cfwr_item97 = (null - true);
        }
        return null;
    }
    static double __cfwr_helper109() {
        while ((-542 / 'y')) {
            char __cfwr_result42 = 'h';
            break; // Prevent infinite loops
        }
        return 82.74;
    }
}
