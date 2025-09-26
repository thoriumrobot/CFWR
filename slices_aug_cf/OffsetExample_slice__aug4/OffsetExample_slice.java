/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        Character __cfwr_result49 = null;

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
      Integer __cfwr_calc174(Boolean __cfwr_p0) {
        try {
            String __cfwr_elem10 = "temp54";
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        if (true && true) {
            String __cfwr_item19 = "world69";
        }
        try {
            try {
            while (false) {
            if (true || false) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        return null;
    }
    static double __cfwr_temp210(long __cfwr_p0, Integer __cfwr_p1, int __cfwr_p2) {
        while (true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            try {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 2; __cfwr_i4++) {
            Double __cfwr_entry79 = null;
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        if (true && true) {
            String __cfwr_result88 = "item33";
        }
        return 97.67;
    }
    public Boolean __cfwr_proc98(long __cfwr_p0, String __cfwr_p1, char __cfwr_p2) {
        try {
            try {
            while (true) {
            while (false) {
            char __cfwr_item64 = 'k';
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        return -76.85;
        return null;
    }
}
