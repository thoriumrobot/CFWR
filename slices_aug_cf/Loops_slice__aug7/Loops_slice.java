/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while ((621 % -2.36f)) {
            while (false) {
            while (true) {
            Integer __cfwr_data62 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
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
      public Object __cfwr_calc231(long __cfwr_p0, Double __cfwr_p1) {
        if (true || (-25.09 - -935)) {
            long __cfwr_temp47 = (null | -383L);
        }
        if (((-338 ^ 51.06) - 132) || true) {
            long __cfwr_entry36 = (73.96 ^ null);
        }
        Integer __cfwr_result47 = null;
        return null;
    }
    Float __cfwr_temp582(boolean __cfwr_p0, short __cfwr_p1, Object __cfwr_p2) {
        return null;
        try {
            try {
            return 'd';
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        try {
            char __cfwr_entry17 = 'F';
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        return null;
    }
    double __cfwr_compute773(int __cfwr_p0) {
        while (((-230L % -374) ^ 792L)) {
            return 'q';
            break; // Prevent infinite loops
        }
        return 83.84;
    }
}
