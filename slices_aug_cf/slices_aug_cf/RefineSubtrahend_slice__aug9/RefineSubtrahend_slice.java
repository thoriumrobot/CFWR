/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        return ((-773 | 42.10) + null);

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
      private static Character __cfwr_helper835(Float __cfwr_p0, Object __cfwr_p1) {
        for (int __cfwr_i42 = 0; __cfwr_i42 < 8; __cfwr_i42++) {
            try {
            try {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 2; __cfwr_i63++) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 4; __cfwr_i81++) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 3; __cfwr_i10++) {
            long __cfwr_result92 = 680L;
        }
        }
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        return null;
    }
    protected static Double __cfwr_func445(Double __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i30 = 0; __cfwr_i30 < 1; __cfwr_i30++) {
            if (true && false) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 2; __cfwr_i56++) {
            return null;
        }
        }
        }
        Long __cfwr_result5 = null;
        if (false || true) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 2; __cfwr_i40++) {
            if (true && (732 | null)) {
            try {
            return null;
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
        }
        }
        return null;
    }
    protected String __cfwr_compute420(int __cfwr_p0, Integer __cfwr_p1) {
        while (false) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 6; __cfwr_i84++) {
            if ((null << (null * true)) || true) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 2; __cfwr_i81++) {
            while (true) {
            while (true) {
            if (((true - -3.43) / 407L) || true) {
            while ((null & (-8.19 % 59.83f))) {
            return -134L;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        return 216;
        try {
            return 30.54f;
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        return "result4";
    }
}
