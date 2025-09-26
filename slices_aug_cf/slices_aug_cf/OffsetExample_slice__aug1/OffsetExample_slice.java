/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        while ((-34.36f / -592)) {
            try {
            return null;
        } catch (Exceptio
        while (false) {
            try {
            try {
            try {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 10; __cfwr_i54++) {
            double __cfwr_val25 = (null << (false + -204));
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
            break; // Prevent infinite loops
        }
n __cfwr_e73) {
            // ignore
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
      protected Object __cfwr_process808(Double __cfwr_p0, int __cfwr_p1, Long __cfwr_p2) {
        while (true) {
            if (true && true) {
            try {
            while (true) {
            while ((-17.98 >> -311L)) {
            while (true) {
            try {
            Double __cfwr_val63 = null;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            while (false) {
            while ((null % null)) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        while (((16.43f * false) * null)) {
            if ((101L % null) && false) {
            try {
            return ('e' >> null);
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        if (true || true) {
            return null;
        }
        return null;
    }
}
