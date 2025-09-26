/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        Integer __cfwr_node4 = null;

    int j = 2;
    int x = a.length;
    int y = x - j;
    a[y] 
        for (int __cfwr_i81 = 0; __cfwr_i81 < 2; __cfwr_i81++) {
            double __cfwr_temp3 = -39.44;
        }
= 0;
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
      byte __cfwr_util803() {
        return null;
        return -57.09;
        if (true && false) {
            long __cfwr_entry59 = (null - 42.08);
        }
        return null;
    }
    protected float __cfwr_handle691(char __cfwr_p0, byte __cfwr_p1, long __cfwr_p2) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 6; __cfwr_i89++) {
            while (false) {
            return 71.95f;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i13 = 0; __cfwr_i13 < 4; __cfwr_i13++) {
            long __cfwr_entry27 = -664L;
        }
        Float __cfwr_result52 = null;
        while (false) {
            if (true || false) {
            return -94.21;
        }
            break; // Prevent infinite loops
        }
        return 40.29f;
    }
    public byte __cfwr_process704() {
        while (true) {
            try {
            short __cfwr_data45 = (56.63 * (null ^ -63.09));
        } catch (Exception __cfwr_e60) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        try {
            if (false && ((-86.30 & null) + 77.71f)) {
            if (false && true) {
            boolean __cfwr_item12 = (null / 'g');
        }
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        return (('6' + -864) ^ 'U');
    }
}
