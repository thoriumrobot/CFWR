/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        for (int __cfwr_i87 = 0; __cfwr_i87 < 5; __cfwr_i87++) {
            for (int __cfwr_i77 = 0; _
        for (int __cfwr_i46 = 0; __cfwr_i46 < 5; __cfwr_i46++) {
            return ((-17L - null) & (true * -638));
        }
_cfwr_i77 < 4; __cfwr_i77++) {
            try {
            return null;
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
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
      Object __cfwr_func762(Long __cfwr_p0) {
        Double __cfwr_node10 = null;
        return null;
        return null;
        while (('b' * 990)) {
            if ((516L ^ '8') || true) {
            return ((null & true) | 4.81f);
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
