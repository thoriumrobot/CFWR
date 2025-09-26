/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        while ((-84.25f & 'e')) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 1; __cfwr_i75++) {

        try {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 2; __cfwr_i96++) {
            if ((966L % (938 ^ 1.24)) && true) {
            return ((-30L + 34.24f) >> 'c');
        }
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
            return null;
        }
            break; // Prevent infinite loops
        }

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
      public static Character __cfwr_process487(double __cfwr_p0, boolean __cfwr_p1) {
        while (true) {
            while ((('J' - -21.08f) - false)) {
            if (false || true) {
            if (true || ('x' | -53.00)) {
            try {
            try {
            return (834 << -540);
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        while (true) {
            try {
            try {
            short __cfwr_data35 = (('d' & '7') << false);
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false && (88.40f >> ('u' >> true))) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 7; __cfwr_i79++) {
            short __cfwr_var88 = (603 % -81.65);
        }
        }
        return null;
    }
    protected boolean __cfwr_handle838(long __cfwr_p0, boolean __cfwr_p1, Long __cfwr_p2) {
        if ((-1.05f >> -41L) || false) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 3; __cfwr_i91++) {
            return null;
        }
        }
        return true;
    }
}
