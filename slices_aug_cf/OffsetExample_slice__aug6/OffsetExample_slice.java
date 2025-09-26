/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        Float __cfwr_data86 = null;

    int j = 2;
    int x = a.length;
    int y = x - j;
    a[y] =
        try {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 1; __cfwr_i99++) {
            try {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 2; __cfwr_i89++) {
            if ((null / true) && true) {
            if (false || true) {
            char __cfwr_data90 = ((-55.08 / null) / null);
        }
        }
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
 0;
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
      private short __cfwr_util86(double __cfwr_p0, Long __cfwr_p1) {
        byte __cfwr_val42 = (-45.78 & 40.31);
        return null;
    }
    protected char __cfwr_util914(Float __cfwr_p0, float __cfwr_p1, Long __cfwr_p2) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 6; __cfwr_i73++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        while (false) {
            short __cfwr_elem66 = null;
            break; // Prevent infinite loops
        }
        try {
            try {
            return 25.70;
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        } catch (Exception __cfwr_e46) {
            // ignore
        }
        return 'Z';
    }
}
