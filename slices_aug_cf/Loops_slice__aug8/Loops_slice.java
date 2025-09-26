/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        try {
    
        Long __cfwr_node44 = null;
        if (false || (594L & 15L)) {
            if (true || false) {
            while ((-54.98f % (false | null))) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 7; __cfwr_i8++) {
            if (true || ((-31.15f | true) | (-212L % -12.53f))) {
            boolean __cfwr_val62 = (-29.69f + (null % null));
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }

    while (flag) {
      // :: error: (unary.increment)
      offset++;
    }
  }

  public void test1b(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (compound.assignment)
      offset += 1;
    }
  }

  public void test1c(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (compound.assignment)
      offset2 += offset;
    }
  }

  public void test2(int[] a, int[] array) {
    int offset = array.length - 1;
    int offset2 = array.length - 1;

    while (flag) {
      offset++;
      offset2 += offset;
    }
    // :: error: (assignment)
    @LTLengthOf("array") int x = offset;
    // :: error: (assignment)
    @LTLengthOf("array") int y = offset2;
  }

  public void test3(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      offset--;
      // :: error: (compound.assignment)
      offset2 -= offset;
    }
  }

  public void test4(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (unary.increment)
      offset++;
      // :: error: (compound.assignment)
      offset += 1;
      // :: error: (compound.assignment)
      offset2 += offset;
    }
  }

  public void test4(int[] src) {
    int patternLength = src.length;
    int[] optoSft = new int[patternLength];
    for (int i = patternLength; i > 0; i--) {}
  }

  public void test5(
      int[] a,
      @LTLengthOf(value = "#1", offset = "-1000") int offset,
      @LTLengthOf("#1") int offset2) {
    int otherOffset = offset;
    while (flag) {
      otherOffset += 1;
      // :: error: (unary.increment)
      offset++;
      // :: error: (compound.assignment)
      offset += 1;
      // :: error: (compound.assignment)
      offset2 += offset;
    }
    // :: error: (assignment)
    @LTLengthOf(value = "#1", offset = "-1000") int x = otherOffset;
      private int __cfwr_helper437(Long __cfwr_p0, Integer __cfwr_p1) {
        try {
            try {
            Integer __cfwr_temp90 = null;
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        if (false && ((false ^ null) >> (63.24 ^ null))) {
            if (true && false) {
            long __cfwr_data70 = -217L;
        }
        }
        if (false || false) {
            char __cfwr_elem46 = 'w';
        }
        if ((-139 / (-58.24 | -50.11)) || false) {
            try {
            if (true && false) {
            for (int __cfwr_i46 = 0; __cfwr_i46 < 3; __cfwr_i46++) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 6; __cfwr_i11++) {
            try {
            while (((true / null) >> 98.08)) {
            boolean __cfwr_elem49 = (-76.57f & null);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        }
        return -396;
    }
    private Integer __cfwr_compute590(int __cfwr_p0) {
        if (((933 << 67) ^ -72.62) || true) {
            if (false && (('q' | -0.05) * true)) {
            Integer __cfwr_result77 = null;
        }
        }
        if (true || true) {
            if (false && false) {
            while ((null ^ (null * -786L))) {
            char __cfwr_temp81 = 'w';
            break; // Prevent infinite loops
        }
        }
        }
        try {
            for (int __cfwr_i66 = 0; __cfwr_i66 < 10; __cfwr_i66++) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 2; __cfwr_i88++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        try {
            return null;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        return null;
    }
}
