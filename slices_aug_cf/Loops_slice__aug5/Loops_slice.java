/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        while (true) {
            try {
            Double __cfwr_obj2 = null;
        } catch (Exception __cfwr_e85) {
            // ignore
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
      public String __cfwr_compute903(String __cfwr_p0, Float __cfwr_p1, byte __cfwr_p2) {
        if ((('B' ^ null) + (null / 136)) && false) {
            return "result48";
        }
        for (int __cfwr_i30 = 0; __cfwr_i30 < 6; __cfwr_i30++) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 5; __cfwr_i49++) {
            while (true) {
            try {
            while (false) {
            try {
            try {
            while (true) {
            try {
            Long __cfwr_var70 = null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        for (int __cfwr_i33 = 0; __cfwr_i33 < 8; __cfwr_i33++) {
            try {
            long __cfwr_node63 = (595 - (null + 26.86));
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        }
        if (false || ((-81.05f / -38.15) / (80 << null))) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 5; __cfwr_i14++) {
            return null;
        }
        }
        return "data50";
    }
    static Boolean __cfwr_handle484(String __cfwr_p0) {
        byte __cfwr_val54 = null;
        if (true || false) {
            while (false) {
            return 4;
            break; // Prevent infinite loops
        }
        }
        float __cfwr_result75 = 24.27f;
        if (true || true) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 10; __cfwr_i42++) {
            if (true || true) {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 10; __cfwr_i53++) {
            Integer __cfwr_val36 = null;
        }
        }
        }
        }
        return null;
    }
    Character __cfwr_proc613(double __cfwr_p0, Integer __cfwr_p1) {
        return (null ^ (-49.39 >> 't'));
        return null;
        return null;
    }
}
