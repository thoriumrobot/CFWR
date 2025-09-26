/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void withConstant(int[] a, @NonNegative int l) {
        if (false || true) {
        Object __cfwr_temp47 = null;

            Boolean __cfwr_obj72 = null;
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
      Double __cfwr_temp182(Double __cfwr_p0, Object __cfwr_p1, long __cfwr_p2) {
        while ((null << 73.48f)) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 6; __cfwr_i43++) {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 5; __cfwr_i99++) {
            while ((null + (-78.13 - null))) {
            while (((-777 & -57.84f) & 36.10f)) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 3; __cfwr_i26++) {
            if ((63.28 - null) && true) {
            float __cfwr_data17 = 50.52f;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        Character __cfwr_item29 = null;
        return "item63";
        return null;
        return null;
    }
    private static Double __cfwr_process88(String __cfwr_p0, boolean __cfwr_p1) {
        try {
            if (false && true) {
            try {
            try {
            return 64.88;
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        for (int __cfwr_i78 = 0; __cfwr_i78 < 2; __cfwr_i78++) {
            try {
            return null;
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        return null;
    }
}
