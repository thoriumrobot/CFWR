/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        try {
            wh
        try {
            if (false || false) {
            return null;
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
ile (true) {
            Float __cfwr_result80 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }

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
      protected static Object __cfwr_proc11(Character __cfwr_p0, Boolean __cfwr_p1) {
        long __cfwr_item57 = 14L;
        return null;
    }
}
