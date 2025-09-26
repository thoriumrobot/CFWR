/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        String __cfwr_entry60 = "hello74";

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
      byte __cfwr_helper176() {
        float __cfwr_val91 = 1.27f;
        if (true || ((-14.01f ^ -762) & true)) {
            while (('m' * null)) {
            if (true && (-484L % -615L)) {
            while (true) {
            try {
            Boolean __cfwr_elem27 = null;
        } catch (Exception __cfwr_e66) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        return ((null % null) >> null);
    }
    protected long __cfwr_helper533(Double __cfwr_p0, Object __cfwr_p1, Character __cfwr_p2) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 1; __cfwr_i51++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        while (true) {
            if (('x' & ('B' - 91.49f)) && false) {
            if (((false << -87.00) ^ false) && false) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 4; __cfwr_i1++) {
            boolean __cfwr_elem33 = false;
        }
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            int __cfwr_entry12 = -516;
            break; // Prevent infinite loops
        }
        return "data1";
        return (-151L + (null - -74.77f));
    }
}
