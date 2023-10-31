function select_column(col) {
    $("#select_column").val(col);
    $("div[name='column-name']").each(function() {
        if($(this).parents("a").hasClass("active")) {
            $(this).parents("a").removeClass("active");
        }
        if($(this).text().trim() == col) {
            $(this).parents("a").addClass("active");
        }
    });

    $.ajax({
        type: 'POST',
        url: '/preprocess/process/detail',
        data: {
            'column': col,
            'csrfmiddlewaretoken': document.querySelector('[name=csrfmiddlewaretoken]').value
        },
        success: function(data) {
            $("#preprocess-detail").html("");
            $("#preprocess-detail").html(data);
        }
    });
}

function select_type() {
    var input = $("input[name='input_for_data_type']");
    var type = input.data('type');
    if( type.indexOf('int') >= 0 ) {
        input.val("숫자형(정수)");
    } else if ( type.indexOf('float') >= 0 ) {
        input.val("숫자형(실수)");
    } else if ( type.indexOf('category') >= 0 ) {
        input.val("카테고리형");
    } else if ( type.indexOf('object') >= 0 ) {
        input.val("문자형");
    } else if ( type.indexOf('datetime') >= 0 ) {
        input.val("시간형");
    }
}

function change_datatype() {
    var column = $("#select_column").val();
    var type = $("#select_for_work").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    $.ajax({
        type: 'POST',
        url: '/preprocess/process/data_type_change',
        data: {
            'column': column,
            'type': type,
            'select_tab': tab_href,
            'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
        },
        success: function(data) {
            $("input[name='input_for_data_type']").data('type', type);
            $("#div_for_dataset_raw").remove();
            $("#div_for_data_graph").remove();
            $("#div_for_data_preview").prepend(data);
        },
        error: function(data) {
            console.log("type change error");
        }
    });
}

function append_item(name) {
    if(name === "delete") {
        var column = $("#column_name").text();
        var choice = '<div class="choices__item choices__item--selectable bg-danger text-light border-danger">변수 삭제' +
        '<button type="button" class="choices__button text-choicesfont border-choicesfont" onclick="remove_item(this);" data-process="delete" column_name='
        + column +' data-work="test">Remove item</button></div>';
        $("#preprocess-choices").append(choice);
        return
    } else {
        var process = $("select[name='select_for_process']").val();
        var work = $("select[name='select_for_work']").val();
        var choice = '<div class="choices__item choices__item--selectable bg-choicesbg text-choicesfont border-choicesbg">' + name
                    + '<button type="button" class="choices__button text-choicesfont border-choicesfont" onclick="remove_item(this);" data-process="' +
                    process + '" data-work="' + work + '">Remove item</button></div>';
        $("#preprocess-choices").append(choice);
    }
}

function remove_item(this_item) {
    var parent_div = $(this_item).parent("div");
    var col = $("#select_column").val();
    var process = $(this_item).data('process');
    var work = $(this_item).data('work');
    var replaceInput = "";

    if ( process == "replace") {
        var temp = $(this_item).closest("div").text().trim().split(" ");
        var input = temp[0].split(":")
        replaceInput = input[1]
    }

    $.ajax({
        type: 'POST',
        url: '/preprocess/process/remove',
        data: {
            'column': col,
            'process': process,
            'work': work,
            'replace_input': replaceInput,
            'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
        },
        success: function(data) {
            parent_div.detach();
            $("#div_for_dataset_raw").remove();
            $("#div_for_data_graph").remove();
            $("#div_for_data_preview").prepend(data);
            item_count();
        },
        error: function(data) {
            console.log("process remove error");
        }
    });
}

function item_count() {
    var count = $("#preprocess-choices").find("div.choices__item").length;
    $("div[name='column-name']").each(function() {
        if ( $(this).text().trim() == $("#select_column").val() ) {
            $(this).closest("a").find("span.badge").text(count);
        }
    });
}

function sorting_text() {
    var $list = $("#data-column-list");
    $list.children().detach().sort(function(a, b) {
        return $(a).text().trim().localeCompare($(b).text().trim());
    }).appendTo($list);
}

function change_preprocess() {
    var value = $("select[name='select_for_process']").val();
    var select_for_work = $("select[name='select_for_work']");

    select_for_work.empty();

    if ( value == 'missing' ) {
        $("p[name='explanation_for_process']").text("데이터의 빈 값, 즉 NaN값을 처리합니다");
        $("#div_for_select_work").show();
        $("#div_for_input").hide();
        $("#div_for_replace").hide();
        var items = { "none": "작업 유형을 선택하세요", "remove": "결측치 제거", "interpolation": "보간값으로 결측치 변환", "mean": "평균값으로 결측치 변환", "input": "변환값 직접 입력" };
        var keys = Object.keys(items);
        for( var i = 0; i < keys.length; i++ ) {
            select_for_work.append("<option value='" + keys[i] + "'>" + items[keys[i]] + "</option>");
        }
    } else if ( value == 'outlier' ) {
        $("p[name='explanation_for_process']").text("정상 데이터와 다른 패턴을 보이는 이상치들 (통계적으로 관측된 다른값들과 멀리 떨어진 값)을 처리합니다");
        $("#div_for_select_work").show();
        $("#div_for_input").hide();
        $("#div_for_replace").hide();
        var items = { "none": "작업 유형을 선택하세요", "remove": "이상치 제거", "minmax": "최소/최대값으로 이상치 변환", "mean": "평균값으로 이상치 변환", "input": "변환값 직접 입력" };
        var keys = Object.keys(items);
        for( var i = 0; i < keys.length; i++ ) {
            select_for_work.append("<option value='" + keys[i] + "'>" + items[keys[i]] + "</option>");
        }
    } else if ( value == 'replace' ) {
        $("p[name='explanation_for_process']").text("기존 데이터 값을 새로운 값으로 입력받아 변환합니다");
        $("#div_for_select_work").hide();
        $("#div_for_input").show();
        $("#div_for_replace").show();
    } else if ( value == 'datatype' ) {
        $("p[name='explanation_for_process']").text("데이터의 유형을 변경합니다");
        $("#div_for_select_work").show();
        $("#div_for_input").hide();
        $("#div_for_replace").hide();
        var data_type = $("#input_for_data_type").val();
        var items = { "int": "숫자형(정수)", "float": "숫자형(실수)", "category": "카테고리형", "object": "문자형", "datetime": "시간형" };
        var keys = Object.keys(items);
        for( var i = 0; i < keys.length; i++ ) {
            if ( data_type == items[keys[i]] ) {
                select_for_work.append("<option value='" + keys[i] + "' selected>" + items[keys[i]] + "</option>");
            } else {
                select_for_work.append("<option value='" + keys[i] + "'>" + items[keys[i]] + "</option>");
            }
        }
    } else if ( value == 'dummy' ) {
        $("p[name='explanation_for_process']").text("기계학습에 적합한 데이터의 형태로 가공하기 위해 수치화할 때, 각 숫자형 카테고리 데이터의 값들이 관계성을 지니지 않도록 가변수화 처리합니다");
        $("#div_for_select_work").hide();
        $("#div_for_input").hide();
        $("#div_for_replace").hide();
    } else if ( value == 'scaler' ) {
        $("p[name='explanation_for_process']").text("특성들의 단위에 영향받지 않고 값의 범위를 비슷하게 만들어 같은 정도의 스케일(중요도)로 반영되도록 처리합니다");
        $("#div_for_select_work").show();
        $("#div_for_input").hide();
        $("#div_for_replace").hide();
        var items = { "none": "작업 유형을 선택하세요", "standard": "StandardScaler", "robust": "RobustScaler", "minmax": "MinMaxScaler", "normal": "Normalizer", "maxabs": "MaxAbsScaler" };
        var keys = Object.keys(items);
        for( var i = 0; i < keys.length; i++ ) {
            select_for_work.append("<option value='" + keys[i] + "'>" + items[keys[i]] + "</option>");
        }
    } else {
        $("p[name='explanation_for_process']").text("");
        $("#div_for_select_work").show();
        $("#div_for_input").hide();
        $("#div_for_replace").hide();
        select_for_work.append("<option value='none' selected>처리 유형을 선택하세요</option>");
    }
}

function process_missing() {
    var column = $("#select_column").val();
    var value = $("select[name='select_for_work']").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    if ( value == "none" ) {
        $("#div_for_replace").hide();
        return false;
    } else if ( value == "input" ) {
        $("#div_for_replace").show();
    } else {
        $("#div_for_replace").hide();
        $.ajax({
            type: 'POST',
            url: '/preprocess/process/process_missing_value',
            data: {
                'column': column,
                'process': value,
                'input_value': "",
                'select_tab': tab_href,
                'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
            },
            success: function(data) {
                $("#div_for_dataset_raw").remove();
                $("#div_for_data_graph").remove();
                $("#div_for_data_preview").prepend(data);
            },
            error: function(data) {
                console.log("process missing value error");
            }
        });
    }
}

function process_outlier() {
    var column = $("#select_column").val();
    var value = $("select[name='select_for_work']").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    if ( value == "none" ) {
        $("#div_for_replace").hide();
        return false;
    } else if ( value == "input" ) {
        $("#div_for_replace").show();
    } else {
        $("#div_for_replace").hide();
        $.ajax({
            type: 'POST',
            url: '/preprocess/process/process_outlier',
            data: {
                'column': column,
                'process': value,
                'input_value': "",
                'select_tab': tab_href,
                'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
            },
            success: function(data) {
                $("#div_for_dataset_raw").remove();
                $("#div_for_data_graph").remove();
                $("#div_for_data_preview").prepend(data);
            },
            error: function(data) {
                console.log("process outlier error");
            }
        });
    }
}

function input_replace_value() {
    var column = $("#select_column").val();
    var work_input = $("#workInput").val();
    var replace_input = $("#replaceInput").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    $.ajax({
        type: 'POST',
        url: '/preprocess/process/replace_value',
        data: {
            'column': column,
            'work_input': work_input,
            'replace_input': replace_input,
            'select_tab': tab_href,
            'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
        },
        success: function(data) {
            $("#div_for_dataset_raw").remove();
            $("#div_for_data_graph").remove();
            $("#div_for_data_preview").prepend(data);
        },
        error: function(data) {
            console.log("process replace value error");
        }
    });
}

function process_dummy() {
    var column = $("#select_column").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    $.ajax({
        type: 'POST',
        url: '/preprocess/process/process_dummy',
        data: {
            'column': column,
            'select_tab': tab_href,
            'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
        },
        success: function(data) {
            $("#div_for_dataset_raw").remove();
            $("#div_for_data_graph").remove();
            $("#div_for_data_preview").prepend(data);
        },
        error: function(data) {
            console.log("process dummy error");
        }
    });
}

function process_scaler() {
    var column = $("#select_column").val();
    var value = $("select[name='select_for_work']").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    $.ajax({
        type: 'POST',
        url: '/preprocess/process/process_scaler',
        data: {
            'column': column,
            'process': value,
            'select_tab': tab_href,
            'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
        },
        success: function(data) {
            $("#div_for_dataset_raw").remove();
            $("#div_for_data_graph").remove();
            $("#div_for_data_preview").prepend(data);
        },
        error: function(data) {
            console.log("process scaler error");
        }
    });

}

function process_replace_input() {
    var value = $("select[name='select_for_process']").val();
    var replace_input = $("#replaceInput").val();
    var column = $("#select_column").val();
    var tab_href = "#statistics";
    $("#nav-graph-tab > li").each(function() {
        if ( $(this).children("a").hasClass("active") ) {
            tab_href = $(this).children("a").attr("href");
        }
    });

    if ( value == 'missing' ) {
        $.ajax({
            type: 'POST',
            url: '/preprocess/process/process_missing_value',
            data: {
                'column': column,
                'process': "input",
                'input_value': replace_input,
                'select_tab': tab_href,
                'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
            },
            success: function(data) {
                $("#div_for_dataset_raw").remove();
                $("#div_for_data_graph").remove();
                $("#div_for_data_preview").prepend(data);
            },
            error: function(data) {
                console.log("process missing value error");
            }
        });
    } else if ( value == 'outlier' ) {
        $.ajax({
            type: 'POST',
            url: '/preprocess/process/process_outlier',
            data: {
                'column': column,
                'process': "input",
                'input_value': replace_input,
                'select_tab': tab_href,
                'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
            },
            success: function(data) {
                $("#div_for_dataset_raw").remove();
                $("#div_for_data_graph").remove();
                $("#div_for_data_preview").prepend(data);
            },
            error: function(data) {
                console.log("process outlier error");
            }
        });
    } else if ( value == 'replace' ) {
        var work_input = $("#workInput").val();

        $.ajax({
            type: 'POST',
            url: '/preprocess/process/process_replace',
            data: {
                'column': column,
                'work_input': work_input,
                'replace_input': replace_input,
                'select_tab': tab_href,
                'csrfmiddlewaretoken': $('[name=csrfmiddlewaretoken]').val()
            },
            success: function(data) {
                $("#div_for_dataset_raw").remove();
                $("#div_for_data_graph").remove();
                $("#div_for_data_preview").prepend(data);
            },
            error: function(data) {
                console.log("process replace error");
            }
        });
    }
}