/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        while (false) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 9; __cfwr_i79++) {
            return -196L;
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
      Integer __cfwr_compute412(int __cfwr_p0, Double __cfwr_p1) {
        if (false || (686L - 168)) {
            while (true) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 7; __cfwr_i75++) {
            if ((-65.57f % null) && true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
        try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 8; __cfwr_i57++) {
            return null;
        }
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        return null;
    }
    public long __cfwr_util55(Float __cfwr_p0, Character __cfwr_p1, Double __cfwr_p2) {
        for (int __cfwr_i28 = 0; __cfwr_i28 < 5; __cfwr_i28++) {
            return null;
        }
        return 121L;
    }
}
