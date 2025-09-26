/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        try {
            short __cfwr_result32 = null;
        } catch (Exception __cfwr_e10) {
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
      protected int __cfwr_compute790(char __cfwr_p0) {
        return -860L;
        if (true && true) {
            if (true || ((-9.32 << true) % null)) {
            if ((-18.29 + 36.84) && false) {
            try {
            String __cfwr_result56 = "item32";
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        }
        }
        for (int __cfwr_i52 = 0; __cfwr_i52 < 10; __cfwr_i52++) {
            if (false && false) {
            while (((970 >> -93.53f) - true)) {
            while (false) {
            while (true) {
            if (((null % -52.01f) + null) || (-66.18f << true)) {
            return null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        if (false || true) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 6; __cfwr_i67++) {
            try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 4; __cfwr_i32++) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 6; __cfwr_i7++) {
            long __cfwr_data10 = ('y' / (null >> -738L));
        }
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        }
        return 782;
    }
    private long __cfwr_aux89(int __cfwr_p0) {
        try {
            Long __cfwr_result58 = null;
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        return 55L;
    }
    protected float __cfwr_handle492(Object __cfwr_p0, short __cfwr_p1) {
        if (true || true) {
            return null;
        }
        for (int __cfwr_i59 = 0; __cfwr_i59 < 1; __cfwr_i59++) {
            if (true && false) {
            for (int __cfwr_i21 = 0; __cfwr_i21 < 4; __cfwr_i21++) {
            Character __cfwr_elem66 = null;
        }
        }
        }
        for (int __cfwr_i32 = 0; __cfwr_i32 < 7; __cfwr_i32++) {
            if (false && (null * -61.64f)) {
            return null;
        }
        }
        boolean __cfwr_node62 = true;
        return ((null - false) - 379);
    }
}
