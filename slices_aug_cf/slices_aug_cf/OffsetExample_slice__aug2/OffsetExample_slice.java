/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        return null;

    int j = 2;
    int x = a.length;
    int y = x - j;
    a[y] = 0;
    for (int i = 0; i < y; i++) {
      a[i + j] = 1;
      a[j + i] = 1;
      a[i + 0] = 1;
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
      a[i + 2 + j] = 1;
    }
  }

  void example3(int @MinLen(2) [] a) {
    int j = 2;
    for (int i = 0; i < a.length - 2; i++) {
      a[i + j] = 1;
    }
  }

  void example4(int[] a, int offset) {
    int max_index = a.length - offset;
    for (int i = 0; i < max_index; i++) {
      a[i + offset] = 0;
    }
  }

  void example5(int[] a, int offset) {
    for (int i = 0; i < a.length - offset; i++) {
      a[i + offset] = 0;
    }
  }

  void test(@IndexFor("#3") int start, @IndexOrHigh("#3") int end, int[] a) {
    if (end > start) {
      // If start == 0, then end - start is end.  end might be equal to the length of a.  So
      // the array access might be too high.
      // ::error: (array.access.unsafe.high)
      a[end - start] = 0;
    }

    if (end > start) {
      a[end - start - 1] = 0;
    }
      private static int __cfwr_calc206(Object __cfwr_p0, Long __cfwr_p1) {
        try {
            while ((true + 336)) {
            while (false) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 5; __cfwr_i68++) {
            return ((-54.01 / null) >> null);
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        return "hello44";
        return -104;
    }
    public static float __cfwr_handle765(boolean __cfwr_p0, Double __cfwr_p1) {
        for (int __cfwr_i90 = 0; __cfwr_i90 < 6; __cfwr_i90++) {
            return null;
        }
        return ('B' % (-829L & 66.15));
    }
    static double __cfwr_compute249(byte __cfwr_p0) {
        while (true) {
            while (false) {
            while ((723L + 'r')) {
            if (((null & 62.97) | (-340L ^ 23.39)) && (null << (null >> -10.19))) {
            if (false || false) {
            if (true || true) {
            while ((null >> -374)) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 6; __cfwr_i30++) {
            try {
            if ((-769L ^ null) || false) {
            if (((-64.04f - 78.09) + -125L) || true) {
            try {
            Float __cfwr_entry6 = null;
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return "hello52";
        if (false && ('Q' % 473L)) {
            try {
            return null;
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        }
        return 75.42;
    }
}
