/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        try {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 10; __cfwr_i90++) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 2; __cfwr_i41++) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 10; __cfwr_i97++) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 3; __cfwr_i59++) {
            return 'y';
        }
        }
        }
        }
        } catch (Exception __cfwr_e5) {
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
      private char __cfwr_process464(Boolean __cfwr_p0) {
        while (true) {
            try {
            if (true && (-10.18f / (true ^ null))) {
            if ((true % null) && true) {
            while (true) {
            try {
            while (false) {
            if (true && true) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 10; __cfwr_i98++) {
            try {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 10; __cfwr_i35++) {
            while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        for (int __cfwr_i98 = 0; __cfwr_i98 < 9; __cfwr_i98++) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 7; __cfwr_i82++) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 9; __cfwr_i84++) {
            if (true && false) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 7; __cfwr_i7++) {
            return null;
        }
        }
        }
        }
        }
        return (null & -43.65);
    }
}
