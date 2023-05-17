var mergingTooltipSlider=document.getElementById("slider_n_estimator"),mergingTooltipSlider02=(mergingTooltipSlider&&(noUiSlider.create(mergingTooltipSlider,{start:[10,100],step:10,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:10,max:500},pips:{mode:"count",values:8}}),mergeTooltips(mergingTooltipSlider,5," - "),valueSettings(mergingTooltipSlider)),document.getElementById("slider_max_depth"));
var mergingTooltipSlider03=document.getElementById("slider_max_iter"),mergingTooltipSlider04=document.getElementById("slider_min_samples_split"), mergingTooltipSlider05=document.getElementById("slider_min_samples_leaf"), mergingTooltipSlider06=document.getElementById("slider_elasticnet"), mergingTooltipSlider07=document.getElementById("slider_reg_param");
function mergeTooltips(e,d,s){var m="rtl"===getComputedStyle(e).direction,c="rtl"===e.noUiSlider.options.direction,u="vertical"===e.noUiSlider.options.orientation,S=e.noUiSlider.getTooltips(),o=e.noUiSlider.getOrigins();Array.from(S).forEach(function(e,i){e&&o[i].appendChild(e)}),e&&e.noUiSlider.on("update",function(e,i,o,r,t){var n=[[]],p=[[]],a=[[]],l=0;S[0]&&(n[0][0]=0,p[0][0]=t[0],a[0][0]=e[0]);for(var g=1;g<t.length;g++)(!S[g]||t[g]-t[g-1]>d)&&(n[++l]=[],a[l]=[],p[l]=[]),S[g]&&(n[l].push(g),a[l].push(e[g]),p[l].push(t[g]));Array.from(n).forEach(function(e,i){for(var o=e.length,r=0;r<o;r++){var t,n,l,g=e[r];r===o-1?(l=0,Array.from(p[i]).forEach(function(e){l+=1e3-e}),t=u?"bottom":"right",n=1e3-p[i][c?0:o-1],l=(m&&!u?100:0)+l/o-n,S[g].innerHTML=a[i].join(s),S[g].style.display="block",S[g].style[t]=l+"%"):S[g].style.display="none"}})})}

function valueSettings(e) {
	e.noUiSlider.on("update", function() {
		var valArray = this.get(0);
		e.parentElement.closest("div.mb-5").querySelectorAll(".form-control").forEach(function(v, n) {
			if ( n == 0 ) {
				v.value = valArray[0];
			} else if ( n == 1 ) {
				v.value = valArray[1];
			}
		});
	});
}

mergingTooltipSlider02&&(noUiSlider.create(mergingTooltipSlider02,{start:[5,15],step:5,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:5,max:40},pips:{mode:"count",values:8}}),mergeTooltips(mergingTooltipSlider02,5," - "),valueSettings(mergingTooltipSlider02));
mergingTooltipSlider03&&(noUiSlider.create(mergingTooltipSlider03,{start:[10,100],step:10,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:10,max:1e3},pips:{mode:"count",values:12}}),mergeTooltips(mergingTooltipSlider03,5, " - "),valueSettings(mergingTooltipSlider03));
mergingTooltipSlider04&&(noUiSlider.create(mergingTooltipSlider04,{start:[1,5],step:1,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:1,max:10},pips:{mode:"count",values:10}}),mergeTooltips(mergingTooltipSlider04,5, " - "),valueSettings(mergingTooltipSlider04));
mergingTooltipSlider05&&(noUiSlider.create(mergingTooltipSlider05,{start:[1,5],step:1,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:1,max:10},pips:{mode:"count",values:10}}),mergeTooltips(mergingTooltipSlider05,5, " - "),valueSettings(mergingTooltipSlider05));
mergingTooltipSlider06&&(noUiSlider.create(mergingTooltipSlider06,{start:[0.01,0.1],step:0.05,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:0.00,max:1.0},pips:{mode:"count",values:11,format: {to: function(value) { return value.toFixed(1);}}}}),mergeTooltips(mergingTooltipSlider06,5, " - "),valueSettings(mergingTooltipSlider06));
mergingTooltipSlider07&&(noUiSlider.create(mergingTooltipSlider07,{start:[0.01,0.1],step:0.05,connect:!0,behaviour:"tap-drag",tooltips:[!0,!0],range:{min:0.00,max:1.0},pips:{mode:"count",values:11,format: {to: function(value) { return value.toFixed(1);}}}}),mergeTooltips(mergingTooltipSlider07,5, " - "),valueSettings(mergingTooltipSlider07));

