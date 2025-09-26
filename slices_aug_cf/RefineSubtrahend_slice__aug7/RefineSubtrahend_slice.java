/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        try {
            wh
        Object __cfwr_var32 = null;
ile (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e12) {
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
      static int __cfwr_util202(Float __cfwr_p0, float __cfwr_p1) {
        if (true || (-423 & true)) {
            Integer __cfwr_val42 = null;
        }
        Float __cfwr_data36 = null;
        for (int __cfwr_i59 = 0; __cfwr_i59 < 5; __cfwr_i59++) {
            while (true) {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 4; __cfwr_i23++) {
            if (true || true) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 7; __cfwr_i11++) {
            while ((-87.87 << 942)) {
            if (((102 / 53) ^ (13.56 & true)) || (816 & 310L)) {
            if (true && true) {
            try {
            return null;
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        return -588;
    }
}
