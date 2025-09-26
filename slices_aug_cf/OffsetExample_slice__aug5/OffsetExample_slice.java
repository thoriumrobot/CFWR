/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        if ((-678L % true) && true) {
            while ((false - 16.65)) {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 3; __cfwr_i24++) {
            return null;
        }
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
      static double __cfwr_proc202() {
        if ((null + 312L) || false) {
            return null;
        }
        Character __cfwr_entry42 = null;
        return (false & -658);
    }
    public static Character __cfwr_compute34(Float __cfwr_p0) {
        for (int __cfwr_i30 = 0; __cfwr_i30 < 10; __cfwr_i30++) {
            return null;
        }
        while (true) {
            Long __cfwr_data13 = null;
            break; // Prevent infinite loops
        }
        while ((null | true)) {
            if (false || false) {
            return (null - null);
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
