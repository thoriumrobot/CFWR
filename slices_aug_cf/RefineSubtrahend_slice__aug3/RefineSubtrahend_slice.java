/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        Character __cfwr_var93 = null;

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
      public static byte __cfwr_temp899(boolean __cfwr_p0, Double __cfwr_p1) {
        if (false && true) {
            if (false || ((true % 96.20f) % 711)) {
            while (false) {
            try {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 5; __cfwr_i8++) {
            try {
            if (('0' << 'J') || true) {
            return "data39";
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        if (('M' ^ 'G') || true) {
            Object __cfwr_var71 = null;
        }
        int __cfwr_temp72 = -790;
        if ((true + (-58.70f ^ null)) && true) {
            while ((-6.19f - -29.27)) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 1; __cfwr_i87++) {
            Boolean __cfwr_item77 = null;
        }
            break; // Prevent infinite loops
        }
        }
        return ('e' | -509L);
    }
    private static Long __cfwr_util885(double __cfwr_p0, Boolean __cfwr_p1) {
        if (((820L - 34.10) | (-7.13 ^ true)) || false) {
            if (true && false) {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 5; __cfwr_i99++) {
            return "temp63";
        }
        }
        }
        if (((-653 * 'N') << 'T') && (('V' << -20.56f) % (-8.59 & -425L))) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 2; __cfwr_i9++) {
            Double __cfwr_val97 = null;
        }
        }
        while (false) {
            return 64.54;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i5 = 0; __cfwr_i5 < 7; __cfwr_i5++) {
            return null;
        }
        return null;
    }
}
