/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void example2(int @MinLen(2) [] a) {
        while (true) {
            while (false) {
            while ((46.89 + 'F')) {
            try 
        for (int __cfwr_i62 = 0; __cfwr_i62 < 9; __cfwr_i62++) {
            while (false) {
            if (true && true) {
            char __cfwr_result42 = 'f';
        }
            break; // Prevent infinite loops
        }
        }
{
            return null;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
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
      protected long __cfwr_func646(String __cfwr_p0, Long __cfwr_p1) {
        try {
            if (true || false) {
            try {
            Boolean __cfwr_result42 = null;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        try {
            try {
            if (('H' + null) || true) {
            try {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 7; __cfwr_i11++) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 8; __cfwr_i94++) {
            String __cfwr_data85 = "item98";
        }
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        return -939L;
    }
}
