/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        if (false 
        for (int __cfwr_i63 = 0; __cfwr_i63 < 2; __cfwr_i63++) {
            return null;
        }
&& true) {
            while (false) {
            if ((('p' << null) - (28.45f * 77.62f)) || (false / 769L)) {
            while (false) {
            for (int __cfwr_i46 = 0; __cfwr_i46 < 1; __cfwr_i46++) {
            while (false) {
            Boolean __cfwr_elem57 = null;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
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
      protected double __cfwr_compute265(double __cfwr_p0, String __cfwr_p1, Integer __cfwr_p2) {
        for (int __cfwr_i18 = 0; __cfwr_i18 < 1; __cfwr_i18++) {
            return null;
        }
        try {
            if (false && true) {
            for (int __cfwr_i67 = 0; __cfwr_i67 < 2; __cfwr_i67++) {
            try {
            try {
            return -761L;
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        return -77.69;
    }
    String __cfwr_helper347() {
        return null;
        for (int __cfwr_i84 = 0; __cfwr_i84 < 6; __cfwr_i84++) {
            for (int __cfwr_i21 = 0; __cfwr_i21 < 3; __cfwr_i21++) {
            return (true ^ (17.83f * -14.62));
        }
        }
        long __cfwr_node91 = 725L;
        return "data86";
    }
    private static Double __cfwr_temp12(Object __cfwr_p0) {
        try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 3; __cfwr_i33++) {
            try {
            double __cfwr_node69 = -18.10;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        return null;
    }
}
