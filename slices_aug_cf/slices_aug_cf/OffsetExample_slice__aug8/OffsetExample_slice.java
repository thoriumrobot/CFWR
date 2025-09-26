/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        if (false && true) {
            while ((227 & ('J' - -7.75f))) {
            return "value61";
            break; // Prevent infinite loops
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
      private Integer __cfwr_proc351(Object __cfwr_p0, boolean __cfwr_p1, Character __cfwr_p2) {
        if (false && ('q' | 13.88f)) {
            while (true) {
            if (false || false) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        if (true && true) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 9; __cfwr_i92++) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 5; __cfwr_i31++) {
            return ((null >> -929) ^ (true ^ null));
        }
        }
        }
        if (false && (170 - 'P')) {
            if ((false + false) && false) {
            Integer __cfwr_entry35 = null;
        }
        }
        return null;
    }
}
