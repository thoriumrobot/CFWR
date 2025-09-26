/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        try {
            double __cfwr_data74 = (-75.30f + false);
        } catch (Exception __cfwr_e61) {
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
      public static String __cfwr_compute624(Long __cfwr_p0) {
        try {
            for (int __cfwr_i19 = 0; __cfwr_i19 < 5; __cfwr_i19++) {
            try {
            if (((null & -72.63) % 'S') || false) {
            Double __cfwr_temp98 = null;
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        for (int __cfwr_i56 = 0; __cfwr_i56 < 5; __cfwr_i56++) {
            try {
            return null;
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        while ((45.77 + null)) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 3; __cfwr_i15++) {
            if ((('h' - true) << 653) && true) {
            if (false || true) {
            if ((42.15f % (890L << 749L)) && ((null & 75.92f) << -23.67f)) {
            return 'M';
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        return "data39";
    }
    public double __cfwr_calc18() {
        if (false || ((null - 'Y') - 14.84f)) {
            while (((-98.72 & '1') ^ null)) {
            return null;
            break; // Prevent infinite loops
        }
        }
        while (true) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 8; __cfwr_i87++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 10; __cfwr_i88++) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 9; __cfwr_i78++) {
            byte __cfwr_val88 = (-999 - (null + 842L));
        }
        }
        if ((null ^ '2') || true) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 5; __cfwr_i77++) {
            if (false && (967 & (90.92f - false))) {
            while (true) {
            while (('o' >> -97.91f)) {
            try {
            return "data91";
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        return ((false | 50.93) + -221L);
    }
    protected Boolean __cfwr_helper819(Boolean __cfwr_p0) {
        return false;
        for (int __cfwr_i8 = 0; __cfwr_i8 < 4; __cfwr_i8++) {
            return -238L;
        }
        while ((181L >> (null << -66.94f))) {
            if (true || false) {
            if ((false + 78.91f) && false) {
            Boolean __cfwr_val49 = null;
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
