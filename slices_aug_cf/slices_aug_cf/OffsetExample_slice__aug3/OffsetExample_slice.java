/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        for (int __cfwr_i57 = 0; __cfwr_i57 < 2; __cfwr_i57++) {
            return null;
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
      static byte __cfwr_proc396(Character __cfwr_p0, Float __cfwr_p1, Double __cfwr_p2) {
        try {
            if ((null + (null << null)) || (-77 & ('J' % 'O'))) {
            try {
            try {
            while (true) {
            long __cfwr_entry35 = (405 << '1');
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        for (int __cfwr_i96 = 0; __cfwr_i96 < 9; __cfwr_i96++) {
            try {
            try {
            if ((null - ('o' ^ 45.35f)) || false) {
            while (true) {
            return "hello76";
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
        return null;
    }
    Object __cfwr_process899(byte __cfwr_p0, byte __cfwr_p1) {
        return null;
        return null;
    }
}
